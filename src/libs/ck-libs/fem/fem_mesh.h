/*
Charm++ Finite Element Framework:
C++ implementation file: Mesh representation and manipulation

This file lists the classes used to represent and manipulate 
Finite Element Meshes inside the FEM framework.

Orion Sky Lawlor, olawlor@acm.org, 1/3/2003
*/
#ifndef __CHARM_FEM_MESH_H
#define __CHARM_FEM_MESH_H

#include "charm-api.h"

#ifndef __CHARM_IDXL_COMM_H
#  include "idxl_comm.h" /*for IDXL_Side */
#endif
#ifndef __CHARM_IDXL_LAYOUT_H
#  include "idxl_layout.h" /*for IDXL_Layout */
#endif

#if defined(WIN32) && defined(max)
#undef max
#endif

// Map the IDXL names to the old FEM names (FIXME: change all references, too)
typedef IDXL_Side FEM_Comm;
typedef IDXL_List FEM_Comm_List;
typedef IDXL_Rec FEM_Comm_Rec;
class IDXL_Chunk;

/// We want the FEM_Comm/IDXL_Side's to be accessible to *both* 
///  FEM routines (via these data structures) and IDXL routines
///  (via an idxl->addStatic registration).  Hence this class, which
///  manages IDXL's view of FEM's data structures.
class FEM_Comm_Holder {
	IDXL comm; //Our communication lists.
	bool registered; //We're registered with IDXL
	IDXL_Comm_t idx; //Our IDXL index, or -1 if we're unregistered
	void registerIdx(IDXL_Chunk *c);
public:
	FEM_Comm_Holder(FEM_Comm *sendComm, FEM_Comm *recvComm);
	void pup(PUP::er &p);
	~FEM_Comm_Holder(void);
	
	/// Return our IDXL_Comm_t, registering with chunk if needed
	inline IDXL_Comm_t getIndex(IDXL_Chunk *c) {
		if (idx==-1) registerIdx(c);
		return idx;
	}
};


/** \addtogroup fem_mesh FEM Framework Library Mesh Representation */
/*\@{*/


/// This datatype is how the framework stores symmetries internally.
///   Each bit of this type describes a different symmetry.
///   There must be enough bits to accomidate several simulatanious symmetries.
typedef unsigned char FEM_Symmetries_t;


/// Describes one kind of symmetry condition
class FEM_Sym_Desc : public PUP::able {
public:
	virtual ~FEM_Sym_Desc();

	/// Apply this symmetry to this location vector
	virtual CkVector3d applyLoc(const CkVector3d &loc) const =0;
	
	/// Apply this symmetry to this relative (vel or acc) vector
	virtual CkVector3d applyVec(const CkVector3d &vec) const =0;
	
	/// Allows Desc's to be pup'd via | operator:
	friend inline void operator|(PUP::er &p,FEM_Sym_Desc &a) {a.pup(p);}
	friend inline void operator|(PUP::er &p,FEM_Sym_Desc* &a) {
		PUP::able *pa=a;  p(&pa);  a=(FEM_Sym_Desc *)pa;
	}
};

/// Describes a linear-periodic (space shift) symmetry:
class FEM_Sym_Linear : public FEM_Sym_Desc {
	CkVector3d shift; //Offset to add to locations
public:
	FEM_Sym_Linear(const CkVector3d &shift_) :shift(shift_) {}
	FEM_Sym_Linear(CkMigrateMessage *m) {}
	
	/// Apply this symmetry to this location vector
	CkVector3d applyLoc(const CkVector3d &loc) const {return loc+shift;}
	
	/// Apply this symmetry to this relative (vel or acc) vector
	virtual CkVector3d applyVec(const CkVector3d &vec) const {return vec;}
	
	virtual void pup(PUP::er &p);
	PUPable_decl(FEM_Sym_Linear);
};

/**
 * Describes all the different kinds of symmetries that apply to
 * this mesh.
 */
class FEM_Sym_List {
	//This lists the different kinds of symmetry
	CkPupAblePtrVec<FEM_Sym_Desc> sym; 
	
	FEM_Sym_List(const FEM_Sym_List &src); //NOT DEFINED: copy constructor
public:
	FEM_Sym_List();
	void operator=(const FEM_Sym_List &src); //Assignment operator
	~FEM_Sym_List();
	
	/// Add a new kind of symmetry to this list, returning
	///  the way objects with that symmetry should be marked.
	FEM_Symmetries_t add(FEM_Sym_Desc *desc);
	
	/// Apply all the listed symmetries to this location
	void applyLoc(CkVector3d *loc,FEM_Symmetries_t sym) const;
	
	/// Apply all the listed symmetries to this relative vector
	void applyVec(CkVector3d *vec,FEM_Symmetries_t sym) const;
	
	void pup(PUP::er &p);
};


/**
 * This is a simple 2D table.  The operations are mostly row-centric.
 */
template <class T>
class BasicTable2d : public CkNoncopyable {
protected:
	int rows; //Number of entries in table
	int cols; //Size of each entry in table
	T *table; //Data in table [rows * cols]
public:
	BasicTable2d(T *src,int cols_,int rows_) 
		:rows(rows_), cols(cols_), table(src) {}
	
	/// "size" of the table is the number of rows:
	inline int size(void) const {return rows;}
	/// Width of the table is the number of columns:
	inline int width(void) const {return cols;}
	
	T *getData(void) {return table;}
	const T *getData(void) const {return table;}
	
/// Element-by-element operations:
	T operator() (int r,int c) const {return table[c+r*cols];}
	T &operator() (int r,int c) {return table[c+r*cols];}
	
/// Row-by-row operations
	//Get a pointer to a row of the table:
	inline T *getRow(int r) {return &table[r*cols];}
	inline const T *getRow(int r) const {return &table[r*cols];}
	inline void getRow(int r,T *dest,T idxBase=0) const {
		const T *src=getRow(r);
		for (int c=0;c<cols;c++) dest[c]=src[c]+idxBase;
	}
	inline T *operator[](int r) {return getRow(r);}
	inline const T *operator[](int r) const {return getRow(r);}
	inline void setRow(int r,const T *src,T idxBase=0) {
		T *dest=getRow(r);
		for (int c=0;c<cols;c++) dest[c]=src[c]-idxBase;
	}
	inline void setRow(int r,T value) {
		T *dest=getRow(r);
		for (int c=0;c<cols;c++) dest[c]=value;
	}
	
/// These affect the entire table:
	void set(const T *src,T idxBase=0) {
		for (int r=0;r<rows;r++) 
		for (int c=0;c<cols;c++)
			table[c+r*cols]=src[c+r*cols]-idxBase;
	}
	void setTranspose(const T *srcT,int idxBase=0) {
		for (int r=0;r<rows;r++) 
		for (int c=0;c<cols;c++)
			table[c+r*cols]=srcT[r+c*rows]-idxBase;
	}
	void get(T *dest,T idxBase=0) const {
		for (int r=0;r<rows;r++) 
		for (int c=0;c<cols;c++)
			dest[c+r*cols]=table[c+r*cols]+idxBase;
	}
	void getTranspose(T *destT,int idxBase=0) const {
		for (int r=0;r<rows;r++) 
		for (int c=0;c<cols;c++)
			destT[r+c*rows]=table[c+r*cols]+idxBase;
	}
	void set(T value) {
		for (int r=0;r<rows;r++) setRow(r,value);
	}
};

/**
 * A heap-allocatable, resizable BasicTable2d.
 * To be stored here, T must not require a copy constructor.
 */
template <class T>
class AllocTable2d : public BasicTable2d<T> {
	int max; //Maximum number of rows that can be used without reallocation
	T fill; //Value to fill uninitialized regions with
	T *allocTable; // the table that I allocated
public:
	AllocTable2d(int cols_=0,int rows_=0,T fill_=0) 
		:BasicTable2d<T>(NULL,cols_,rows_), max(0), fill(fill_),allocTable(NULL)
	{
		if (this->rows>0) allocate(this->rows);
	}
	~AllocTable2d() {delete[] allocTable;}
	/// Make room for this many rows
	void allocate(int rows_) { 
		allocate(this->width(),rows_);
	}
	/// Make room for this many cols & rows
	void allocate(int cols_,int rows_,int max_=0) { 
		if (cols_==this->cols && rows_<max) {
			//We have room--just update the size:
			this->rows=rows_;
			return;
		}
		if (max_==0) { //They gave no suggested size-- pick one:
			if (rows_==this->rows+1) //Growing slowly: grab a little extra
				max_=10+rows_+(rows_>>2); //  125% plus 10
			else // for a big change, just go with the minimum needed: 
				max_=rows_;
		}

		int oldRows=this->rows;
		this->cols=cols_;
		this->rows=rows_;
		this->max=max_;
		this->table=new T[max*this->cols];
		//Preserve old table entries (FIXME: assumes old cols is unchanged)
		int copyRows=0;
		if (allocTable!=NULL) { 
			copyRows=oldRows;
			if (copyRows>max) copyRows=max;
			memcpy(this->table,allocTable,sizeof(T)*this->cols*copyRows);
			delete[] allocTable;
		}else{
			for (int r=copyRows;r<max;r++)
				this->setRow(r,fill);
		}
		allocTable = this->table;
	}
	
	/// Pup routine and operator|:
	void pup(PUP::er &p) {
		p|this->rows; p|this->cols;
		if (this->table==NULL) allocate(this->rows);
		p(this->table,this->rows*this->cols); //T better be a basic type, or this won't compile!
	}

	void pupSingle(PUP::er &p, int pupindx) {
	  p|this->table[pupindx];
	}

	friend void operator|(PUP::er &p,AllocTable2d<T> &t) {t.pup(p);}

	/// Add a row to the table (by analogy with std::vector):
	T *push_back(void) {
		if (this->rows>=max) 
		{ //Not already enough room for the new row:
			int newMax=max+(max/4)+16; //Grow 25% longer
			allocate(this->cols,this->rows,newMax);
		}
		this->rows++;
		return getRow(this->rows-1);
	}

	/** to support replacement of attribute data by user 
			supplied data
			error checks have been performed at FEM_ATTRIB
	*/
	void register_data(T *user,int len,int max_){
		if(allocTable != NULL){
			delete [] allocTable;
			allocTable = NULL;
		}	
		this->table = user;
		this->rows = len;
		max = max_;
	}
};


class FEM_Entity; // Forward declaration
class FEM_Mesh; 


/// Return the human-readable version of this entity code, like "FEM_NODE".
///  storage, which must be at least 80 bytes long, is used for
///  non-static names, like the user tag "FEM_ELEM+2".
CDECL const char *FEM_Get_entity_name(int entity,char *storage);

/// Return the human-readable version of this attribute code, like "FEM_CONN".
///  storage, which must be at least 80 bytes long, is used for
///  non-static names, like the user tag "FEM_DATA+7".
CDECL const char *FEM_Get_attr_name(int attr,char *storage);



/**
 * Describes an FEM entity's "attribute"--a user-visible, user-settable
 * 2D table.  Common FEM_Attributes include: user data associated with nodes,
 *  the element-to-node connectivity array, etc.
 */
class FEM_Attribute {
	FEM_Entity *e; //Owning entity (to get length, etc.)
	FEM_Attribute *ghost; // Ghost attribute, which has the same width and datatype as us (or 0)
	int attr; // My attribute code (e.g., FEM_DATA+7, FEM_CONN, etc.)
	
	int width; //Number of columns in our table of data (initially unknown)
	int datatype; //Datatype of entries (initially unknown)
	bool allocated; //True if subclass allocate has been called.
	
	//Abort with a nice error message saying: 
	// Our <field> was previously set to <cur>; it cannot now be <next>
	void bad(const char *field,bool forRead,int cur,int next,const char *caller) const;
	
protected:
	/**
	 * Allocate storage for at least length width-item records of type datatype.
	 * This routine is called after all three parameters are set,
	 * as a convenience for subclasses. 
	 */
	virtual void allocate(int length,int width,int datatype) =0;
public:
	FEM_Attribute(FEM_Entity *owner_,int myAttr_);
	virtual void pup(PUP::er &p);
	virtual void pupSingle(PUP::er &p, int pupindx);
	virtual ~FEM_Attribute();
	
	/// Install this attribute as our ghost:
	inline void setGhost(FEM_Attribute *ghost_) { ghost=ghost_;}
	
	/// Return true if we're a ghost
	inline bool isGhost(void) const { return ghost!=NULL; }
	
	/// Return our attribute code
	inline int getAttr(void) const {return attr;}

	inline FEM_Entity *getEntity(void) {return e;}
	
	/// Return the number of rows in our table of data (0 if unknown).
	/// This value is obtained directly from our owning Entity.
	int getLength(void) const;
	int getRealLength(void) const;

	int getMax();
	
	/// Return the number of columns in our table of data (0 if unknown)
	inline int getWidth(void) const { return width<0?0:width; }
	inline int getRealWidth(void) const { return width; }
	
	/// Return our FEM_* datatype (-1 if unknown)
	inline int getDatatype(void) const { return datatype; }
	
	/**
	 * Set our length (number of rows, or records) to this value.
	 * Default implementation calls setLength on our owning entity
	 * and tries to call allocate().
	 */
	void setLength(int l,const char *caller="");
	
	/**
	 * Set our width (number of values per row) to this value.
	 * The default implementation sets width and tries to call allocate().
	 */
	void setWidth(int w,const char *caller="");
	
	/**
	 * Set our datatype (e.g., FEM_INT, FEM_DOUBLE) to this value.
	 * The default implementation sets width and tries to call allocate().
	 */
	void setDatatype(int dt,const char *caller="");
	
	/**
	 * Copy our width and datatype from this attribute.
	 * The default implementation calls setWidth and setDatatype.
	 * which should be enough for virtually any attribute.
	 */
	virtual void copyShape(const FEM_Attribute &src);
	
	/// Check if all three of length, width, and datatype are set,
	///  but we're not yet allocated.  If so, call allocate; else ignore.
	void tryAllocate(void);
	
	/// Our parent's length has changed: reallocate our storage
	inline void reallocate(void) { allocated=false; tryAllocate(); }
	
	/// Return true if we've already had storage allocated.
	inline bool isAllocated(void) const {return allocated;}
	
	/**
	 * Set our data to these (typically user-supplied, unchecked) values.
	 * Subclasses normally override this method as:
	 *    virtual void set( ...) {
	 *       super::set(...);
	 *       copy data from src.
	 *    }
	 */
	virtual void set(const void *src, int firstItem,int length, 
		const IDXL_Layout &layout, const char *caller);
	
	/**
	 * Extract this quantity of user data.  Length and layout are
	 * parameter checked by the default implementation.
	 * Subclasses normally override this method as:
	 *    virtual void get( ...) {
	 *       super::get(...);
	 *       copy data to dest.
	 *    }
	 */
	virtual void get(void *dest, int firstItem,int length,
		const IDXL_Layout &layout, const char *caller) const;
	
	/// Copy everything associated with src[srcEntity] into our dstEntity.
	virtual void copyEntity(int dstEntity,const FEM_Attribute &src,int srcEntity) =0;

	/** Register this user data for this attributre 
			Length, layout etc are checked by the default implementaion
	*/

	virtual void register_data(void *dest, int length,int max,
		const IDXL_Layout &layout, const char *caller);
	
};
PUPmarshall(FEM_Attribute)


/**
 * Describes a single table of user data associated with an entity.
 * Since the data can be of any type, it is stored as chars.
 */
class FEM_DataAttribute : public FEM_Attribute {
	typedef FEM_Attribute super;
	AllocTable2d<unsigned char> *char_data; //Non-NULL for getDatatype==FEM_BYTE
	AllocTable2d<int> *int_data; //Non-NULL for getDatatype==FEM_INT
	AllocTable2d<float> *float_data; //Non-NULL for getDatatype==FEM_FLOAT
	AllocTable2d<double> *double_data; //Non-NULL for getDatatype==FEM_DOUBLE
protected:
	virtual void allocate(int length,int width,int datatype);
public:
	FEM_DataAttribute(FEM_Entity *owner,int myAttr);
	virtual void pup(PUP::er &p);
	virtual void pupSingle(PUP::er &p, int pupindx);
	~FEM_DataAttribute();
	
	AllocTable2d<unsigned char> &getChar(void) {return *char_data;}
	const AllocTable2d<unsigned char> &getChar(void) const {return *char_data;}
	
	AllocTable2d<int> &getInt(void) {return *int_data;}
	const AllocTable2d<int> &getInt(void) const {return *int_data;}
	
	AllocTable2d<double> &getDouble(void) {return *double_data;}

	
	virtual void set(const void *src, int firstItem,int length, 
		const IDXL_Layout &layout, const char *caller);
	
	virtual void get(void *dest, int firstItem,int length, 
		const IDXL_Layout &layout, const char *caller) const;
	
	virtual void register_data(void *dest, int length,int max,
		const IDXL_Layout &layout, const char *caller);

	
	/// Copy src[srcEntity] into our dstEntity.
	virtual void copyEntity(int dstEntity,const FEM_Attribute &src,int srcEntity);

	/*used during refining to extrapolate the values of a node */
	void interpolate(int A,int B,int D,double frac);
	void interpolate(int *iNodes,int rNode,int k);
};
PUPmarshall(FEM_DataAttribute)

/**
 * This table maps an entity to a set of integer indices.
 * The canonical example of this is the element-node connectivity array.
 */
class FEM_IndexAttribute : public FEM_Attribute {
public:
	/// Checks incoming indices for validity.
	class Checker {
	public:
		virtual ~Checker();
		
		/**
		 * Check this (newly set) row of our table for validity.
		 * You're expected to abort or throw or exit if something is wrong.
		 */
		virtual void check(int row,const BasicTable2d<int> &table,
			const char *caller) const =0;
	};
private:
	typedef FEM_Attribute super;
	AllocTable2d<int> idx;
	Checker *checker; //Checks indices (or NULL). This attribute will destroy this object
protected:
	virtual void allocate(int length,int width,int datatype);
public:
	FEM_IndexAttribute(FEM_Entity *owner,int myAttr, Checker *checker_=NULL);
	virtual void pup(PUP::er &p);
	virtual void pupSingle(PUP::er &p, int pupindx);
	~FEM_IndexAttribute();
	
	AllocTable2d<int> &get(void) {return idx;}
	const AllocTable2d<int> &get(void) const {return idx;}
	
	virtual void set(const void *src, int firstItem,int length, 
		const IDXL_Layout &layout, const char *caller);
	
	virtual void get(void *dest, int firstItem,int length,
		const IDXL_Layout &layout, const char *caller) const;

	virtual void register_data(void *dest, int length, int max,
		const IDXL_Layout &layout, const char *caller);
	
	/// Copy src[srcEntity] into our dstEntity.
	virtual void copyEntity(int dstEntity,const FEM_Attribute &src,int srcEntity);
};
PUPmarshall(FEM_IndexAttribute)

/*
	This table maps an entity to a list of integer indices 
	of unknown length.
	The node to element adjacency array is an example, where a node
	is mapped to a list of element indices of unknown length.
*/
class FEM_VarIndexAttribute : public FEM_Attribute{
public:
 class ID{
 	//type is negative for ghost elements
	//id refers to the index in the entity list
 	public:
 		int type;
		int id;
		ID(){
			type=-1;
			id = -1;
		};
		ID(int _type,int _id){
		  if(_id < 0) {
			type = -(_type+1);
			id = FEM_To_ghost_index(_id);
		  }
		  else {
			type = _type;
			id = _id;
		  }
		};
		bool operator ==(const ID &rhs)const {
			return (type == rhs.type) && (id == rhs.id);
		}
		virtual void pup(PUP::er &p){
			p | type;
			p | id;
		};

		static ID createNodeID(int type,int node){
		  ID temp(type, node);
		  return temp;
		}
		int getSignedId() {
		  if(type<0){
			return FEM_From_ghost_index(id);
		  }
		  else return id;
		}
 };
private:
	typedef FEM_Attribute super;
	CkVec<CkVec<ID> > idx;
	int oldlength;
protected:
	virtual void allocate(int _length,int _width,int _datatype){
	  if(_length > oldlength){
	    setWidth(1,"allocate"); //there is 1 vector per entity 
	    oldlength = _length*2;
	    idx.reserve(oldlength);
	    for(int i=idx.size();i<oldlength;i++){
	      CkVec<ID> tempVec;
	      idx.insert(i,tempVec);
	    }
	  }
	};
public:
	FEM_VarIndexAttribute(FEM_Entity *owner,int myAttr);
	~FEM_VarIndexAttribute(){};
	virtual void pup(PUP::er &p);
	virtual void pupSingle(PUP::er &p, int pupindx);
	CkVec<CkVec<ID> > &get(){return idx;};
	const CkVec<CkVec<ID> > &get() const {return idx;};
	
	virtual void set(const void *src,int firstItem,int length,
		const IDXL_Layout &layout,const char *caller);

	virtual void get(void *dest, int firstItem,int length,
		const IDXL_Layout &layout, const char *caller) const;
	
	virtual void copyEntity(int dstEntity,const FEM_Attribute &src,int srcEntity);

	int findInRow(int row,const ID &data);
	
	void print();
};


class l2g_t;
/**
 * Describes an entire class of "entities"--nodes, elements, or sparse 
 *  data records. Basically consists of a length and a set of
 *  FEM_Attributes. 
 */
class FEM_Entity {
	typedef CkVec<FEM_Symmetries_t> sym_t;
	int length; // Number of entities of our type currently in the mesh
	int max;    // Maximum number of entities of our type in the mesh that will be allowed
	FEM_Mesh_alloc_fn resize; // should be a function pointer to the actual resize function later
	void *args; // arguments to the resize function
	
	/**
	 * This is our main list of attributes-- everything about each of 
	 * our entities is in this list of attributes.  This list is searched
	 * by our "lookup" method and maintained by our subclasses "create" 
	 * method and calls to our "add" method.
	 * 
	 * It's a little funky having the superclass keep pointers to subclass
	 * objects (like element connectivity), but very nice to be able to
	 * easily loop over everything associated with an entity.
	 */
	CkVec<FEM_Attribute *> attributes;
	
	/**
	 * Coordinates of each entity, from FEM_COORD.
	 * Datatype is always FEM_DOUBLE, width is always 2 or 3.
	 *  If NULL, coordinates are unknown.
	 */
	FEM_DataAttribute *coord; 
	void allocateCoord(void);
	
	/**
	 * Symmetries of each entity, from FEM_SYMMETRIES.  This bitvector per
	 * entity indicates which symmetry conditions the entity belongs to.
	 * Datatype is always FEM_BYTE (same as FEM_Symmetries_t), width 
	 * is always 1.  If NULL, all the symmetries are 0.
	 */
	FEM_DataAttribute *sym; 
	void allocateSym(void);
	
	/**
	 * Global numbers of each entity, from FEM_GLOBALNO.
	 * If NULL, the global numbers are unknown.
	 */
	FEM_IndexAttribute *globalno;
	void allocateGlobalno(void);
	
	/*
		used to allocate the integer array for storing the boundary
		values associated with an entity. 
	*/
	void allocateBoundary();

	/// Mesh sizing attribute for elements
	/** Specifies a double edge length target for the mesh at each 
	    element; used in adaptivity algorithms */
	FEM_DataAttribute *meshSizing;
	void allocateMeshSizing(void);

	/*
	    used to allocate the char array for storing whether each entity is valid
		When a node/element is deleted the flag in the valid table is set to 0.
		Additionally, we keep track of the first and last occurence in the array of
		invalid indices. This should make searching for slots to reuse quicker.
	*/
	FEM_DataAttribute *valid;
	unsigned int first_invalid, last_invalid;
	
protected:
	/**
	 * lookup of this attribute code has failed: check if it needs
	 * to be demand-created.  Subclasses should override this method
	 * to recognize a request for, and add their own attributes;
	 * otherwise call the default implementation.
	 *
	 * Every call to create must result in either a call to add()
	 * or a call to the superclass; but not both.
	 *
	 * Any entity with optional fields, that are created on demand, will
	 * have to override this method.  Entities with fixed fields that are
	 * known beforehand should just call add() from their constructor.
	 */
	virtual void create(int attr,const char *caller);
	
	/// Add this attribute to this kind of Entity.
	/// This superclass is responsible for eventually deleting the attribute.
	/// This class also attaches the ghost attribute, so be sure to 
	///   call add before manipulating the attribute.
	void add(FEM_Attribute *attribute);
 public:

	FEM_Entity *ghost; // Our ghost entity type, or NULL if we're the ghost
		
	FEM_Comm ghostSend; //Non-ghosts we send out (only set for real entities)
	FEM_Comm ghostRecv; //Ghosts we recv into (only set for ghost entities)

	FEM_Entity(FEM_Entity *ghost_); //Default constructor
	void pup(PUP::er &p);
	virtual ~FEM_Entity();
	
	/// Return true if we're a ghost
	bool isGhost(void) const {return ghost==NULL;}
	
	/// Switch from this, a real entity, to the ghosts:
	FEM_Entity *getGhost(void) {return ghost;}
	const FEM_Entity *getGhost(void) const {return ghost;}
	
	/// Return the number of entities of this type
	inline int size(void) const {return length==-1?0:length;}
	inline int realsize(void) const {return length;}

	// return the maximum size 
	inline int getMax() { if(max > 0) return max; else return length;}
	
	/// Return the human-readable name of this entity type, like "node"
	virtual const char *getName(void) const =0;
	
	/// Copy all our attributes' widths and data types from this entity.
	void copyShape(const FEM_Entity &src);
	
	/** 
	 *  The user is setting this many entities.  This reallocates
	 * all existing attributes to make room for the new entities.
	 */
	void setLength(int newlen);

	/** Support for registration API 
	 *  Set the current length and maximum length for this entity. 
	 *  If the current length exceeds the maximum length a resize
	 *  method is called .
	*/
	void setMaxLength(int newLen,int newMaxLen,void *args,FEM_Mesh_alloc_fn fn);
	
	/// Copy everything associated with src[srcEntity] into our dstEntity.
	/// dstEntity must have already been allocated, e.g., with setLength.
	void copyEntity(int dstEntity,const FEM_Entity &src,int srcEntity);

	/// Add room for one more entity, with initial values from src[srcEntity],
	/// and return the new entity's index.
	int push_back(const FEM_Entity &src,int srcEntity);
	
	/**
	 * Find this attribute (from an FEM_ATTR code) of this entity, or 
	 * create the entity (using the create method below) or abort if it's
	 * not found.
	 */
	FEM_Attribute *lookup(int attr,const char *caller);

	
	/**
	 * Get a list of the attribute numbers for this entity.
	 */
	int getAttrs(int *attrs) const {
		int len=attributes.size();
		for (int i=0;i<len;i++) 
			attrs[i]=attributes[i]->getAttr();
		return len;
	}
	
	/**
	 * Allocate or Modify the FEM_IS_VALID attribute data
	 */
	void allocateValid();
	void set_valid(unsigned int idx, bool isNode);
	void set_invalid(unsigned int idx, bool isNode);
	int is_valid(unsigned int idx);
	unsigned int count_valid();
	unsigned int get_next_invalid(FEM_Mesh *m, bool isNode, bool isGhost);
	virtual bool hasConn(unsigned int idx)=0;
	/**
	 * Set the coordinates for a single item
	 */
	void set_coord(int idx, double x, double y);
	void set_coord(int idx, double x, double y, double z);

	/**expose the attribute vector for refining 
		. breaks modularity but more efficient
	*/
	CkVec<FEM_Attribute *>* getAttrVec(){
		return &attributes;
	}
	
	// Stupidest possible coordinate access
	inline FEM_DataAttribute *getCoord(void) {return coord;}
	inline const FEM_DataAttribute *getCoord(void) const {return coord;}
	
	//Symmetry array access:
	const FEM_Symmetries_t *getSymmetries(void) const {
		if (sym==NULL) return NULL;
		else return (const FEM_Symmetries_t *)sym->getChar()[0];
	}
	FEM_Symmetries_t getSymmetries(int r) const { 
		if (sym==NULL) return FEM_Symmetries_t(0);
		else return sym->getChar()(r,0);
	}
	void setSymmetries(int r,FEM_Symmetries_t s);
	
	//Global numbering array access
	bool hasGlobalno(void) const {return globalno!=0;}
	int getGlobalno(int r) const {
		if (globalno==0) return -1; // Unknown global number
		return globalno->get()(r,0);
	}
	void setGlobalno(int r,int g);
	void setAscendingGlobalno(void);
	void setAscendingGlobalno(int base);
	void copyOldGlobalno(const FEM_Entity &e);

	// Mesh sizing array access
	bool hasMeshSizing(void) const {return meshSizing!=0;}
	double getMeshSizing(int r); 
	void setMeshSizing(int r,double s);
	void setMeshSizing(double *sf);
	
	//Ghost comm. list access
	FEM_Comm &setGhostSend(void) { return ghostSend; }
	const FEM_Comm &getGhostSend(void) const { return ghostSend; }
	FEM_Comm &setGhostRecv(void) { 
		if (ghost==NULL) return ghostRecv;
		else return ghost->ghostRecv; 
	}
	const FEM_Comm &getGhostRecv(void) const { return ghost->ghostRecv; }
	FEM_Comm_Holder ghostIDXL; //IDXL interface
	
	void addVarIndexAttribute(int code){
		FEM_VarIndexAttribute *varAttribute = new FEM_VarIndexAttribute(this,code);
		add(varAttribute);
	}
	
	void print(const char *type,const IDXL_Print_Map &map);
};
PUPmarshall(FEM_Entity)

// Now that we have FEM_Entity, we can define attribute lenth, as entity length
inline int FEM_Attribute::getLength(void) const { return e->size(); }
inline int FEM_Attribute::getRealLength(void) const { return e->realsize(); }
inline int FEM_Attribute::getMax(){ return e->getMax();}

/**
 * Describes a set of FEM Nodes--the FEM_NODE entity type. 
 * Unlike elements, nodes have no connectivity; but they do have
 * special shared-nodes communications and a "primary" flag.
 */

class FEM_Elem;
class FEM_Node : public FEM_Entity {
	typedef FEM_Entity super;

	/**
	 * primary flag, from FEM_NODE_PRIMARY, indicates this chunk "owns" this node.
	 * Datatype is always FEM_BYTE (we need an FEM_BIT!), width is always 1,
	 * since there's only one such flag per node.
	 */
	FEM_DataAttribute *primary; 
	void allocatePrimary(void);
	
	void allocateElemAdjacency();
	void allocateNodeAdjacency();

	FEM_VarIndexAttribute *elemAdjacency; // stores the node to element adjacency vector 
	FEM_VarIndexAttribute *nodeAdjacency; // stores the node to node adjacency vector 
	typedef FEM_VarIndexAttribute::ID var_id;
protected:
	virtual void create(int attr,const char *caller);
public:
	FEM_Comm shared; //Shared nodes
	FEM_Comm_Holder sharedIDXL; //IDXL interface to shared nodes
	
	FEM_Node(FEM_Node *ghost_);
	void pup(PUP::er &p);
	~FEM_Node();
	
	virtual const char *getName(void) const;
	
	inline bool getPrimary(int nodeNo) const {
		if (primary==NULL) return true; //Everything must be primary
		else return primary->getChar()(nodeNo,0);
	}
	inline void setPrimary(int nodeNo,bool isPrimary) {
		if (primary==NULL) allocatePrimary();
		primary->getChar()(nodeNo,0)=isPrimary;
	}
	void fillElemAdjacencyTable(int type,const FEM_Elem &elem);
	void setElemAdjacency(int type,const FEM_Elem &elem);
	void fillNodeAdjacency(const FEM_Elem &elem);
	void setNodeAdjacency(const FEM_Elem &elem);
	void fillNodeAdjacencyForElement(int node,int nodesPerElem,const int *conn,FEM_VarIndexAttribute *adjacencyAttr);
	bool hasConn(unsigned int idx);
	void print(const char *type,const IDXL_Print_Map &map);
};
PUPmarshall(FEM_Node)

/**
 * Describes one kind of FEM elements--the FEM_ELEM entity type.
 * Elements are typically the central user-visible object in a FEM
 * computation.
 */
class FEM_Elem:public FEM_Entity {
	typedef FEM_Entity super;
protected:
	typedef AllocTable2d<int> conn_t;

	int tuplesPerElem;

	// The following are attributes that will commonly be used:
	FEM_IndexAttribute *conn;                // FEM_CONN attribute: element-to-node mapping 
	FEM_IndexAttribute *elemAdjacency;       // FEM_ELEM_ELEM_ADJACENCY attribute
	FEM_IndexAttribute *elemAdjacencyTypes;  // FEM_ELEM_ELEM_ADJ_TYPES attribute

public:

	FEM_Elem(const FEM_Mesh &mesh_, FEM_Elem *ghost_);
	void pup(PUP::er &p);
	~FEM_Elem();
	
	virtual const char *getName(void) const;
	
	/// Directly access our connectivity table:
	inline conn_t &setConn(void) {return conn->get();}
	inline const conn_t &getConn(void) const {return conn->get();}
	
	void print(const char *type,const IDXL_Print_Map &map);

	void create(int attr,const char *caller);

	void allocateElemAdjacency();

	FEM_IndexAttribute *getElemAdjacency(){return elemAdjacency;}

	// Backward compatability routines:
	int getConn(int elem,int nodeNo) const {return conn->get()(elem,nodeNo);}
	int getNodesPer(void) const {return conn->get().width();}
	int *connFor(int i) {return conn->get().getRow(i);}
	const int *connFor(int i) const {return conn->get().getRow(i);}
	void connIs(int i,const int *src) {conn->get().setRow(i,src);}
	bool hasConn(unsigned int idx);
};
PUPmarshall(FEM_Elem)



/**
 * FEM_Sparse describes a set of records of sparse data that are all the
 * same size and all associated with the same number of nodes.
 * Sparse data is associated with some subset of the nodes in the mesh,
 * and gets copied to every chunk that has all those nodes.  The canonical
 * use of sparse data is to describe boundary conditions.
 */
class FEM_Sparse : public FEM_Elem {
	typedef FEM_Elem super;
	typedef AllocTable2d<int> elem_t;
	
	/**
	 * elem, from FEM_SPARSE_ELEM, is an optional (that is, possibly NULL) 
	 * array which changes the partitioning of sparse entities: if non-NULL,
	 * sparse entity t lives on the same chunk as FEM_ELEM+elem[2*t]
	 * local number elem[2*t+1].
	 *
	 * This attribute's width is always 2.
	 */
	FEM_IndexAttribute *elem; //FEM_SPARSE_ELEM attribute
	void allocateElem(void);
	const FEM_Mesh &mesh; //Reference to enclosing mesh, for error checking
protected:
	virtual void create(int attr,const char *caller);
public:
	FEM_Sparse(const FEM_Mesh &mesh_, FEM_Sparse *ghost_);
	void pup(PUP::er &p);
	virtual ~FEM_Sparse();
	
	virtual const char *getName(void) const;
	
	/// Return true if we have an element partitioning table
	bool hasElements(void) const {return elem!=NULL;}
	
	/// Directly access our element partitioning table (e.g., for re-numbering)
	inline elem_t &setElem(void) {return elem->get();}
	inline const elem_t &getElem(void) const {return elem->get();}
};
PUPmarshall(FEM_Sparse)

/** Describes a user function to pup a piece of mesh data 
*/
class FEM_Userdata_pupfn {
	FEM_Userdata_fn fn;
	void *data;
public:
	FEM_Userdata_pupfn(FEM_Userdata_fn fn_,void *data_)
		:fn(fn_), data(data_) {}
	/// Call user's pup routine using this PUP::er
	void pup(PUP::er &p) {
		(fn)((pup_er)&p,data);
	}
};

/** Describes one piece of generic unassociated mesh data.
*/
class FEM_Userdata_item {
	CkVec<char> data; ///< Serialized data from user's pup routine
public:
	int tag; //User-assigned identifier
	FEM_Userdata_item(int tag_=-1) {tag=tag_;}
	
	/// Return true if we have stored data.
	bool hasStored(void) const {return data.size()!=0;}
	
	/// Store this userdata inside us:
	void store(FEM_Userdata_pupfn &f) {
		data.resize(PUP::size(f));
		PUP::toMemBuf(f,&data[0],data.size());
	}
	/// Extract this userdata from our stored copy:
	void restore(FEM_Userdata_pupfn &f) {
		PUP::fromMemBuf(f,&data[0],data.size());
	}
	
	/// Save our stored data to this PUP::er
	void pup(PUP::er &p) {
		p|tag;
		p|data;
	}
};

/** Describes all the unassociated data in a mesh. 
*/
class FEM_Userdata_list {
	CkVec<FEM_Userdata_item> list;
public:
	FEM_Userdata_item &find(int tag) {
		for (int i=0;i<list.size();i++)
			if (list[i].tag==tag)
				return list[i];
		// It's not in the list-- add it.
		list.push_back(FEM_Userdata_item(tag));
		return list[list.size()-1];
	}
	int size(){
		return list.size();
	}
	void pup(PUP::er &p) {p|list;}
};


void FEM_Index_Check(const char *caller,const char *entityType,int type,int maxType);
void FEM_Is_NULL(const char *caller,const char *entityType,int type);

/**
 * This class describes several different types of a certain kind 
 * of entity.  For example, there might be a FEM_Entity_Types<FEM_Elem>
 * that lists the different kinds of element.
 *
 * This class exists to provide a nice "demand-creation" semantics,
 * where the user assigns array indices (the e in FEM_ELEM+e),
 * so we don't know that we're setting the first copy when we set it.
 *
 * It's not clear this class has any right to exist--it should either be
 * folded into FEM_Mesh or generalized into a "userNumberedVec" or some such.
 */
template <class T>
class FEM_Entity_Types {
	CkVec<T *> types; // Our main storage for different entity types
	const FEM_Mesh &mesh;
	const char *name; //FEM_SPARSE or FEM_ELEM, or some such.

public:
	FEM_Entity_Types(const FEM_Mesh &mesh_,const char *name_) 
		:mesh(mesh_), name(name_) {}
	void pup(PUP::er &p) { 
		// Can't just use p|types, because T has a funky constructor:
		int n=types.size();
		p|n;
		for (int i=0;i<n;i++) {
			int isNULL=0;
			if (!p.isUnpacking()) isNULL=(types[i]==NULL);
			p|isNULL;
			if (!isNULL) set(i,"pup").pup(p);
		}
	}
	~FEM_Entity_Types() {
		for (int i=0;i<types.size();i++)
			if (types[i]) delete types[i];
	}
	
	/// Return the number of different entity types
	inline int size(void) const {return types.size();}
	
	/// Return a read-only copy of this type, or else abort if type isn't set.
	const T &get(int type,const char *caller="") const {
		FEM_Index_Check(caller,name,type,types.size());
		const T *ret=types[type];
		if (ret==NULL) FEM_Is_NULL(caller,name,type);
		return *ret;
	}
	
	/// Return true if we have a type t, and false otherwise
	bool has(int type) const { 
		if (type>=types.size()) return false;
		return types[type]!=NULL; 
	}
	
	/// Return a writable copy of this type, calling new T(mesh) if it's not there
	T &set(int type,const char *caller="") {
		if (type<0) FEM_Index_Check(caller,name,type,types.size());
		while (types.size()<=type) types.push_back(NULL); //Make room for new type
		if (types[type]==NULL) { //Have to allocate a new T:
			T *ghost=new T(mesh,NULL);
			types[type]=new T(mesh,ghost);
		}
		return *types[type];
	}
	
	/// Read-only and write-only operator[]'s:
	inline T &operator[] (int type) { return set(type); }
	inline const T &operator[] (int type) const { return get(type); }

};

// Map fortran element (>=1) or node (0) marker to C version (>=1, -1)
inline int zeroToMinusOne(int i) {
	if (i==0) return -1;
	return i;
}




/**
 * This class describes all the nodes and elements in
 * a finite-element mesh or submesh.
 */
class FEM_ElemAdj_Layer;
class femMeshModify;
class FEM_Mesh : public CkNoncopyable {
  /// The symmetries in the mesh
  FEM_Sym_List symList;
  bool m_isSetting;
  femMeshModify *fmMM;
  
  void checkElemType(int elType,const char *caller) const;
  void checkSparseType(int uniqueID,const char *caller) const; 
  
  FEM_ElemAdj_Layer* lastElemAdjLayer;

 public:
  void setFemMeshModify(femMeshModify *m);
  
  FEM_Mesh();
  void pup(PUP::er &p); //For migration
  ~FEM_Mesh();
  
  /// The nodes in this mesh:
  FEM_Node node; 
  
  /// The different element types in this mesh:
  FEM_Entity_Types<FEM_Elem> elem;
  
  /// The different sparse types in this mesh:
  FEM_Entity_Types<FEM_Sparse> sparse;
  
  /// The unassociated user data for this mesh:
  FEM_Userdata_list udata;
  
  /// The symmetries that apply to this mesh:
  void setSymList(const FEM_Sym_List &src) {symList=src;}
  const FEM_Sym_List &getSymList(void) const {return symList;}
  
  /// Set up the "shape" of our fields-- the number of element types,
  /// the datatypes for user data, etc--based on this mesh.
  void copyShape(const FEM_Mesh &src);

  // Get the fem mesh modification object associated with this mesh or partition  
  femMeshModify *getfmMM();

  //Return this type of element, given an element type
  FEM_Entity &setCount(int elTypeOrMinusOne) {
    if (elTypeOrMinusOne==-1) return node;
    else return elem[chkET(elTypeOrMinusOne)];
  }
  const FEM_Entity &getCount(int elTypeOrMinusOne) const {
    if (elTypeOrMinusOne==-1) return node;
    else return elem[chkET(elTypeOrMinusOne)];
  }
  FEM_Elem &setElem(int elType) {return elem[chkET(elType)];}
  const FEM_Elem &getElem(int elType) const {return elem[chkET(elType)];}
  int chkET(int elType) const; //Check this element type-- abort if it's bad
  
  /// Look up this FEM_Entity type in this mesh, or abort if it's not valid.
  FEM_Entity *lookup(int entity,const char *caller);
  const FEM_Entity *lookup(int entity,const char *caller) const;
  
  /// Set/get direction control:
  inline bool isSetting(void) const {return m_isSetting;}
  void becomeSetting(void) {m_isSetting=true;}
  void becomeGetting(void) {m_isSetting=false;}
  
  int nElems() const //Return total number of elements (of all types)
    {return nElems(elem.size());}
  /// Return the total number of elements before type t
  int nElems(int t) const;
  /// Return the "global" number of element elNo of type elType.
  int getGlobalElem(int elType,int elNo) const;
  /// Set our global numbers as 0...n-1 for nodes, elements, and sparse
  void setAscendingGlobalno(void);
  ///	The global numbers for elements runs across different types
  void setAbsoluteGlobalno();
  
  void copyOldGlobalno(const FEM_Mesh &m);
  void print(int idxBase);//Write a human-readable description to CkPrintf
  /// Extract a list of our entities:
  int getEntities(int *entites);
  
  
  
  /********** New methods ***********/
  /*
    This method creates the mapping from a node to all the elements that are 
    incident on it . It assumes the presence of one layer of ghost nodes that
    share a node.
  */
  void createNodeElemAdj();
  void createNodeNodeAdj();
  void createElemElemAdj();
  
  FEM_ElemAdj_Layer *getElemAdjLayer(void);
  
  // Terry's adjacency accessors & modifiers

  //  ------- Element-to-element: preserve initial ordering relative to nodes
  /// Place all of element e's adjacent elements in neighbors; assumes
  /// neighbors allocated to correct size
  void e2e_getAll(int e, int *neighborss, int etype=0);
  /// Given id of element e, return the id of the idx-th adjacent element
  int e2e_getNbr(int e, short idx, int etype=0);
  /// Given id of element e and id of another element nbr, return i such that
  /// nbr is the i-th element adjacent to e
  int e2e_getIndex(int e, int nbr, int etype=0);
  /// Set the element adjacencies of element e to neighbors; assumes neighbors 
  /// has the correct size
  void e2e_setAll(int e, int *neighbors, int etype=0);
  /// Set the idx-th element adjacent to e to be newElem
  void e2e_setIndex(int e, short idx, int newElem, int etype=0);
  /// Find element oldNbr in e's adjacent elements and replace with newNbr
  void e2e_replace(int e, int oldNbr, int newNbr, int etype=0);
  /// Remove all neighboring elements in adjacency
  void e2e_removeAll(int e, int etype=0);


  //  ------- Element-to-node: preserve initial ordering
  /// Place all of element e's adjacent nodes in adjnodes; assumes
  /// adjnodes allocated to correct size
  void e2n_getAll(int e, int *adjnodes, int etype=0);
  /// Given id of element e, return the id of the idx-th adjacent node
  int e2n_getNode(int e, short idx, int etype=0);
  /// Given id of element e and id of a node n, return i such that
  /// n is the i-th node adjacent to e
  short e2n_getIndex(int e, int n, int etype=0);
  /// Set the node adjacencies of element e to adjnodes; assumes adjnodes 
  /// has the correct size
  void e2n_setAll(int e, int *adjnodes, int etype=0);
  /// Set the idx-th node adjacent to e to be newNode
  void e2n_setIndex(int e, short idx, int newNode, int etype=0);
  /// Find node oldNode in e's adjacent ndoes and replace with newNode
  void e2n_replace(int e, int oldNode, int newNode, int etype=0);
  /// Replace all entries with -1
  void e2n_removeAll(int e, int etype=0);


  //  ------- Node-to-node
  /// Place all of node n's adjacent nodes in adjnodes and the resulting 
  /// length of adjnodes in sz; assumes adjnodes is not allocated, but sz is
  void n2n_getAll(int n, int **adjnodes, int *sz);
  /// Adds newNode to node n's node adjacency list
  void n2n_add(int n, int newNode);
  /// Removes oldNode from n's node adjacency list
  void n2n_remove(int n, int oldNode);
  /// Finds oldNode in n's node adjacency list, and replaces it with newNode
  void n2n_replace(int n, int oldNode, int newNode);
  /// Remove all nodes from n's node adjacency list
  void n2n_removeAll(int n);
  /// Is queryNode in node n's adjacency vector?
  int n2n_exists(int n, int queryNode);

  //  ------- Node-to-element
  /// Place all of node n's adjacent elements in adjelements and the resulting 
  /// length of adjelements in sz; assumes adjelements is not allocated, 
  /// but sz is
  void n2e_getAll(int n, int **adjelements, int *sz);
  /// Adds newElem to node n's element adjacency list
  void n2e_add(int n, int newElem);
  /// Removes oldElem from n's element adjacency list
  void n2e_remove(int n, int oldElem);
  /// Finds oldElem in n's element adjacency list, and replaces it with newElem
  void n2e_replace(int n, int oldElem, int newElem);
  /// Remove all elements from n's element adjacency list
  void n2e_removeAll(int n);

  /// Get an element on edge (n1, n2) where n1, n2 are chunk-local
  /// node numberings and result is chunk-local element; return -1 in case 
  /// of failure
  int getElementOnEdge(int n1, int n2);

  /// Get two elements adjacent to both n1 and n2
  void get2ElementsOnEdge(int n1, int n2, int *result_e1, int *result_e2) ;





}; 
PUPmarshall(FEM_Mesh)
FEM_Mesh *FEM_Mesh_lookup(int fem_mesh,const char *caller);
FEM_Entity *FEM_Entity_lookup(int fem_mesh,int entity,const char *caller);
FEM_Attribute *FEM_Attribute_lookup(int fem_mesh,int entity,int attr,const char *caller);

void FEM_Mesh_data_layout(int fem_mesh,int entity,int attr, 	
  	void *data, int firstItem,int length, const IDXL_Layout &layout);

//registration internal api
void FEM_Register_array_layout(int fem_mesh,int entity,int attr,void *data,int firstItem,const IDXL_Layout &layout);
void FEM_Register_entity_impl(int fem_mesh,int entity,void *args,int len,int max,FEM_Mesh_alloc_fn fn);
/// Reassemble split chunks into a single mesh
FEM_Mesh *FEM_Mesh_assemble(int nchunks,FEM_Mesh **chunks);

FILE *FEM_openMeshFile(const char *prefix,int chunkNo,int nchunks,bool forRead);
FEM_Mesh *FEM_readMesh(const char *prefix,int chunkNo,int nChunks);
void FEM_writeMesh(FEM_Mesh *m,const char *prefix,int chunkNo,int nChunks);

/*\@}*/


#endif
