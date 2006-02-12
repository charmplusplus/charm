/* 
ParFUM_internals.h

This file should contain ALL required header code that is 
internal to ParFUM, but should not be visible to users. This
includes all the old fem_mesh.h type files.

*/

#ifndef __PARFUM_INTERNALS_H
#define __PARFUM_INTERNALS_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>

#include "charm-api.h" /* for Fortran name mangling FTN_NAME */
#include "ckvector3d.h"
#include "tcharm.h"
#include "charm++.h"
#include "converse.h" /* for CmiGetArg */
#include "cklists.h"
#include "mpi.h"
#include "pup_mpi.h"
#define checkMPI pup_checkMPI

#include "idxl_layout.h"
#include "idxl.h"
#include "idxl_comm.h"

#include "fem.decl.h"
#include "msa/msa.h"
#include "cklists.h"
#include "pup.h"

#include "ParFUM.decl.h"

/* USE of this extern may be a BUG */
extern CProxy_femMeshModify meshMod;

#define MAX_CHUNK 1000000000

CtvExtern(FEM_Adapt_Algs *, _adaptAlgs);

/*
Charm++ Finite Element Framework:
C++ implementation file: Mesh representation and manipulation

This file lists the classes used to represent and manipulate 
Finite Element Meshes inside the FEM framework.

Orion Sky Lawlor, olawlor@acm.org, 1/3/2003
*/

// Map the IDXL names to the old FEM names (FIXME: change all references, too)
typedef IDXL_Side FEM_Comm;
typedef IDXL_List FEM_Comm_List;
typedef IDXL_Rec FEM_Comm_Rec;

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
	~AllocTable2d() {if(allocTable != NULL){delete[] allocTable;}}
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
				setRow(r,fill);
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
	void register_data(T *user,int len,int max){
		if(allocTable != NULL){
			delete [] allocTable;
			allocTable = NULL;
		}	
		this->table = user;
		this->rows = len;
		max = max;
	}
};

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
PUPmarshall(FEM_Attribute);


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
PUPmarshall(FEM_DataAttribute);

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
PUPmarshall(FEM_IndexAttribute);

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
PUPmarshall(FEM_Entity);

// Now that we have FEM_Entity, we can define attribute lenth, as entity length
inline int FEM_Attribute::getLength(void) const { return e->size(); }
inline int FEM_Attribute::getRealLength(void) const { return e->realsize(); }
inline int FEM_Attribute::getMax(){ return e->getMax();}

/**
 * Describes a set of FEM Nodes--the FEM_NODE entity type. 
 * Unlike elements, nodes have no connectivity; but they do have
 * special shared-nodes communications and a "primary" flag.
 */

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
PUPmarshall(FEM_Node);

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
PUPmarshall(FEM_Elem);



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
PUPmarshall(FEM_Sparse);

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
PUPmarshall(FEM_Mesh);
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




/*Charm++ Finite Element Framework:
C++ implementation file

This is the main internal implementation file for FEM.
It includes all the other headers, and contains quite
a lot of assorted leftovers and utility routines.

Orion Sky Lawlor, olawlor@acm.org, 9/28/00
*/

/** \addtogroup fem_impl FEM Framework Library Implementation */
/*\@{*/

/* A stupid, stupid number: the maximum value of user-defined "elType" fields. 
  This should be dynamic, so any use of this should be considered a bug.
*/
#define FEM_MAX_ELTYPE 20

// Verbose abort routine used by FEM framework:
void FEM_Abort(const char *msg);
void FEM_Abort(const char *caller,const char *sprintf_msg,int int0=0,int int1=0,int int2=0);


/*This class describes a local-to-global index mapping, used in FEM_Print.
The default is the identity mapping.*/
class l2g_t {
public:
	//Return the global number associated with this local element
	virtual int el(int t,int localNo) const {return localNo;}
	//Return the global number associated with this local node
	virtual int no(int localNo) const {return localNo;}
};

/* Map (user-assigned) numbers to T's */
template <class T>
class NumberedVec {
	CkPupPtrVec<T, CkPupAlwaysAllocatePtr<T> > vec;
	
public:
	//Extend the vector to have up to this element
	void makeLonger(int toHaveElement)
	{
		int oldSize=vec.size(), newSize=toHaveElement+1;
		if (oldSize>=newSize) return; //Nothing to do
		vec.resize(newSize);
		for (int j=oldSize;j<newSize;j++)
			vec[j]=new T;
	}
	//Reinitialize element i:
	void reinit(int doomedEl) {
		vec[doomedEl].destroy();
		vec[doomedEl]=new T;
	}
	
	int size(void) const {return vec.size();}
	
	//Same old bracket operators, but return the actual object, not a pointer:
	T &operator[](int i) {
		if (i>=vec.size()) makeLonger(i);
		return *( vec[i] );
	}
	const T &operator[](int i) const {return *( vec[i] );}
	
	void pup(PUP::er &p) {
		vec.pup(p);
	}
	friend void operator|(PUP::er &p,NumberedVec<T> &v) {v.pup(p);}
};


//Smart pointer-to-new[]'d array-of-T
template <class T>
class ArrayPtrT : public CkNoncopyable {
	T *sto;
public:
	ArrayPtrT() {sto=NULL;}
	ArrayPtrT(int *src) {sto=src;}
	~ArrayPtrT() {if (sto) delete[] sto;}
	void operator=(T *src) {
		if (sto) delete[] sto;
		sto=src;
	}
	operator T *(void) {return sto;}
	operator const T *(void) const {return sto;}
	T& operator[](int i) {return sto[i];}
	const T& operator[](int i) const {return sto[i];}
};
typedef ArrayPtrT<int> intArrayPtr;



/* Unmarshall into a heap-allocated copy */
template<class T>
class marshallNewHeapCopy {
	T *cur;
public:
	//Used on send side:
	marshallNewHeapCopy(T *readFrom) :cur(readFrom) {}
	marshallNewHeapCopy(const marshallNewHeapCopy &h) :cur(h.cur) {}
	marshallNewHeapCopy(void) { //Used on recv side:
		cur=new T;
	}
	
	void pup(PUP::er &p) {
		cur->pup(p);
	}
	operator T *() {return cur;}
	friend void operator|(PUP::er &p,marshallNewHeapCopy<T> &h) {h.pup(p);}
};
typedef marshallNewHeapCopy<FEM_Mesh> marshallMeshChunk;


/// Keeps a list of dynamically-allocated T objects,
///  indexed by a user-carried, persistent "int".
template<class T>
class FEM_T_List {
	CkPupPtrVec<T> list; // Vector of T's
protected:
	int FIRST_DT; // User index of first T
	int size(void) const {return list.size();}
	
	/// If this isn't a valid, allocated index, abort.
	inline void check(int l,const char *caller) const {
		if (l<FIRST_DT || l>=FIRST_DT+list.size() || list[l-FIRST_DT]==NULL) 
			badIndex(l,caller);
	}
	
	void badIndex(int l,const char *caller) const {
		if (l<FIRST_DT || l>FIRST_DT+list.size()) bad(l,0,caller);
		else bad(l,1,caller);
	}
public:
	FEM_T_List(int FIRST_DT_) :FIRST_DT(FIRST_DT_) {}
	virtual ~FEM_T_List() {}
	void pup(PUP::er &p) { p|list; }
	
	/// This routine is called when we're passed an invalid T index.
	virtual void bad(int l,int bad_code,const char *caller) const =0;
	
	/// Insert a new T (allocated with "new"), returning the user index:
	int put(T *t) {
		for (int i=0;i<list.size();i++) 
			if (list[i]==NULL) {
				list[i]=t;
				return FIRST_DT+i;
			}
		int ret=list.size();
		list.push_back(t);
		return FIRST_DT+ret;
	}
	
	/// Get this T given its user index.
	inline T *lookup(int l,const char *caller) const {
		check(l,caller);
		return list[l-FIRST_DT];
	}
	
	/// Free this T
	void destroy(int l,const char *caller) {
		check(l,caller);
		list[l-FIRST_DT].destroy();
	}
	
	/// Clear all stored T's:
	void empty(void) {
		for (int i=0;i<list.size();i++) list[i].destroy();
	}
};
class FEM_Mesh_list : public FEM_T_List<FEM_Mesh> {
	typedef FEM_T_List<FEM_Mesh> super;
public:
	FEM_Mesh_list() :super(FEM_MESH_FIRST) { }
	
	virtual void bad(int l,int bad_code,const char *caller) const;
};

#define CHK(p) do{if((p)==0)CkAbort("FEM>Memory Allocation failure.");}while(0)

/**
  FEM global data object.  Keeps track of the global
  list of meshes, and the default read and write meshes.
  
  This class was once an array element, back when the 
  FEM framework was built directly on Charm++.
  
  There's only one of this object per thread, and it's
  kept in a thread-private variable.
*/
class FEM_chunk 
{
public:
  FEM_Mesh_list meshes; ///< Global list of meshes.
  int default_read; ///< Index of default read mesh.
  int default_write; ///< Index of default write mesh.
  
  /// Default communicator to use
  FEM_Comm_t defaultComm;

  /// Global index (rank) in default communicator
  int thisIndex;

#ifdef CMK_OPTIMIZE /* Skip the check, for speed. */
  inline void check(const char *where) { }
#else /* Do an extensive self-check */
  void check(const char *where);
#endif

private:
  CkVec<int> listTmp;//List of local entities, for ghost list exchange
 
  void initFields(void);

 public:
  FEM_chunk(FEM_Comm_t defaultComm_);
  FEM_chunk(CkMigrateMessage *msg);
  void pup(PUP::er &p);
  ~FEM_chunk();
  
  /// Return this thread's single static FEM_chunk instance:
  static FEM_chunk *get(const char *caller);
  
  inline FEM_Mesh *lookup(int fem_mesh,const char *caller) {
     return meshes.lookup(fem_mesh,caller);
  }

  inline FEM_Mesh *getMesh(const char *caller) 
  	{return meshes.lookup(default_read,caller);}
  inline FEM_Mesh *setMesh(const char *caller) 
  	{return meshes.lookup(default_write,caller);}

  void print(int fem_mesh,int idxBase);
  int getPrimary(int nodeNo) { return getMesh("getPrimary")->node.getPrimary(nodeNo); }
  const FEM_Comm &getComm(void) {return getMesh("getComm")->node.shared;}

  // Basically everything below here should be moved to IDXL:
  void exchangeGhostLists(int elemType,int inLen,const int *inList,int idxbase);
  void recvList(int elemType,int fmChk,int nIdx,const int *idx);
  const CkVec<int> &getList(void) {return listTmp;}
  void emptyList(void) {listTmp.length()=0;}
  
  void reduce_field(int idxl_datatype, const void *nodes, void *outbuf, int op);
  void reduce(int idxl_datatype, const void *inbuf, void *outbuf, int op);
  void readField(int idxl_datatype, void *nodes, const char *fname);  
};

/// Describes a single layer of ghost elements.
class FEM_Ghost_Layer : public CkNoncopyable {
public:
	int nodesPerTuple; //Number of shared nodes needed to connect elements
	bool addNodes; //Add ghost nodes to the chunks
	class elemGhostInfo {
	public:
		bool add; //Add this kind of ghost element to the chunks
		int tuplesPerElem; //# of tuples surrounding this element
		intArrayPtr elem2tuple; //The tuples around this element [nodesPerTuple * tuplesPerElem]
		elemGhostInfo(void) {add=false;tuplesPerElem=0;}
		~elemGhostInfo(void) {}
		void pup(PUP::er &p) {//CkAbort("FEM> Shouldn't call elemGhostInfo::pup!\n");
		}
	};
	elemGhostInfo elem[FEM_MAX_ELTYPE];
	virtual void pup(PUP::er &p){
		p | nodesPerTuple;
		p | addNodes;
		for(int i=0;i<FEM_MAX_ELTYPE;i++){
			p | elem[i].add;
			p | elem[i].tuplesPerElem;
			if(elem[i].tuplesPerElem == 0){
				continue;
			}
			int *arr;
			if(p.isUnpacking()){
				arr = new int[nodesPerTuple*elem[i].tuplesPerElem];
			}else{
				arr = elem[i].elem2tuple;
			}
			p(arr,nodesPerTuple*elem[i].tuplesPerElem);
			if(p.isUnpacking()){
				elem[i].elem2tuple = arr;
			}
		}
	}
};

/// Describes a set of required adjacent elements for this kind of element,
///  stored as an explicit adjacency list.
class FEM_Ghost_Stencil {
	/// Our element type.
	int elType;
	/// Number of elements we describe.
	int n;
	/// If true, add ghost nodes as well as elements
	bool addNodes;
	
	/// Last adjacency entry (plus one), indexed by element.
	///  That is, element i's data is at [ends[i-1],ends[i])
	intArrayPtr ends;
	
	/** Adjacency entries for each element.
	  Stored as a series of pairs: elType, elNum.
	  The first pair for element i starts at
	  	2*(ends[i-1]) 
	  the last pair for element i starts at
	  	2*(ends[i]-1)
	  This array then has, in total, 2*ends[n-1] elements.
	*/
	intArrayPtr adj;
public:
	/**
	  Create a stencil with this number of elements, 
	  and these adjecent elements.
	*/
	FEM_Ghost_Stencil(int elType_, int n_,bool addNodes_,
		const int *ends_,
		const int *adj_,
		int idxBase);
	
	/// Make sure this stencil makes sense for this mesh.
	void check(const FEM_Mesh &mesh) const;
	
	/// Return the type of element we describe
	inline int getType(void) const {return elType;}
	
	/**
	  Return a pair consisting of the i'th element's
	  j'th neighbor: the return value's first int is an element type,
	  the second int is an element number.  Returns NULL if i doesn't
	  have a j'th neighbor.
	*/
	inline const int *getNeighbor(int i,int j) const {
		int start=0;
		if (i>0) start=ends[i-1];
		if (j>=ends[i]-start) return 0;
		return &adj[2*(start+j)];
	}
	
	inline bool wantNodes(void) const {return addNodes;}
};

/// Describes a way to grow a set of ghosts.
class FEM_Ghost_Region {
public:	
	FEM_Ghost_Layer *layer;
	FEM_Ghost_Stencil *stencil;
	
	FEM_Ghost_Region() {layer=0; stencil=0;}
	FEM_Ghost_Region(FEM_Ghost_Layer *l) {layer=l; stencil=0;}
	FEM_Ghost_Region(FEM_Ghost_Stencil *s) {layer=0; stencil=s;}
};


//Accumulates all symmetries of the mesh before splitting:
class FEM_Initial_Symmetries; /*Defined in symmetries.C*/

/// Describes all the data needed for partitioning a mesh.
class FEM_Partition : public CkNoncopyable {
	/// Maps element number to (0-based) chunk number, allocated with new[]
	int *elem2chunk;
	
	/// Describes the different regions of ghost elements:
	CkVec<FEM_Ghost_Region> regions;
	FEM_Ghost_Layer *lastLayer;
	
	/// Describes the problem domain's spatial symmetries.
	FEM_Initial_Symmetries *sym;
public:
	FEM_Partition();
	~FEM_Partition();
	
// Manipulate partitioning information
	void setPartition(const int *elem2chunk, int nElem, int idxBase);
	const int *getPartition(FEM_Mesh *src,int nChunks) const;
	
// Manipulate ghost layers
	FEM_Ghost_Layer *addLayer(void) {
		lastLayer=new FEM_Ghost_Layer();
		regions.push_back(lastLayer);
		return lastLayer;
	}
	FEM_Ghost_Layer *curLayer(void) {
		if (lastLayer==0) CkAbort("Must call FEM_Add_ghost_layer before FEM_Add_ghost_elem\n");
		return lastLayer;
	}
	
// Manipulate ghost stencils
	void addGhostStencil(FEM_Ghost_Stencil *s) {
		regions.push_back(s);
		lastLayer=0;
	}
	void markGhostStencilLayer(void) {
		regions.push_back(FEM_Ghost_Region());
	}
	
// Read back ghost regions
	int getRegions(void) const {return regions.size();}
	const FEM_Ghost_Region &getRegion(int regNo) const {return regions[regNo];}
	
// Manipulate spatial symmetries:
	void setSymmetries(int nNodes_,int *new_can,const int *sym_src);
	void addLinearPeriodic(int nFaces_,int nPer,
		const int *facesA,const int *facesB,int idxBase,
		int nNodes_,const CkVector3d *nodeLocs);
	const int *getCanon(void) const;
	const FEM_Symmetries_t *getSymmetries(void) const;
	const FEM_Sym_List &getSymList(void) const;


};
// Access the latest partition:
FEM_Partition &FEM_curPartition(void);

//Declare this at the start of every API routine:
#define FEMAPI(routineName) TCHARM_API_TRACE(routineName,"fem")
//#define FEMAPI(routineName) printf("%s\n", routineName);


/*Partition this mesh's elements into n chunks,
 writing each element's 0-based chunk number to elem2chunk.
*/
void FEM_Mesh_partition(const FEM_Mesh *mesh,int nchunks,int *elem2chunk);

/*A way to stream out partitioned chunks of a mesh.
  By streaming, we can send the chunks as they are built,
  dramatically reducing the memory needed by the framework.
*/
class FEM_Mesh_Output {
 public:
	virtual ~FEM_Mesh_Output() {} /*<- for whining compilers*/
	//Transfer ownership of this mesh chunk
	virtual void accept(int chunkNo,FEM_Mesh *msg) =0;
};

/*After partitioning, create a sub-mesh for each chunk's elements,
including communication lists between chunks.
*/
void FEM_Mesh_split(FEM_Mesh *mesh,int nchunks,
	const FEM_Partition &partition,FEM_Mesh_Output *out);


//Make a new[]'d copy of this (len-entry) array, changing the index as spec'd
int *CkCopyArray(const int *src,int len,int indexBase);


// Isaac's stuff:
// Describes Element Faces. For use with finding element->element adjacencies
// based on FEM_Ghost_Layer
class FEM_ElemAdj_Layer : public CkNoncopyable {
 public:
  int initialized;
  int nodesPerTuple; //Number of shared nodes for a pair of elements
 
 class elemAdjInfo {
  public:
    //  int recentElType; // should not be here, but if it is it should be pup'ed
    int tuplesPerElem; //# of tuples surrounding this element, i.e. number of faces on an element
    intArrayPtr elem2tuple; //The tuples around this element [nodesPerTuple * tuplesPerElem]
    elemAdjInfo(void) {/*add=false;*/tuplesPerElem=0;}
    ~elemAdjInfo(void) {}
    void pup(PUP::er &p) {//CkAbort("FEM> Shouldn't call elemGhostInfo::pup!\n");
    }
  };
 
 elemAdjInfo elem[FEM_MAX_ELTYPE];

 FEM_ElemAdj_Layer() {initialized=0;}

  virtual void pup(PUP::er &p){
    p | nodesPerTuple;
	p | initialized;
    for(int i=0;i<FEM_MAX_ELTYPE;i++){
      p | elem[i].tuplesPerElem;
	  if(elem[i].tuplesPerElem == 0){
	continue;
      }
      int *arr;
      if(p.isUnpacking()){
	arr = new int[nodesPerTuple*elem[i].tuplesPerElem];
      }else{
	arr = elem[i].elem2tuple;
      }
      p(arr,nodesPerTuple*elem[i].tuplesPerElem);
      if(p.isUnpacking()){
	elem[i].elem2tuple = arr;
      }
    }
  }
};


/*\@}*/

// End impl.h





/*
	File containing the data structures, function declaration 
	and msa array declarations used during parallel partitioning
	of the mesh.
	Author Sayantan Chakravorty
	05/30/2004
*/


//#define PARALLEL_DEBUG

#ifdef PARALLEL_DEBUG
#define DEBUG(x) x
#else
#define DEBUG(x)
#endif

#define MESH_CHUNK_TAG 3000

template <class T, bool PUP_EVERY_ELEMENT=true >
class DefaultListEntry {
public:
    inline void accumulate(T& a, const T& b) { a += b; }
    // identity for initializing at start of accumulate
    inline T getIdentity() { return T(); }
    inline bool pupEveryElement(){ return PUP_EVERY_ELEMENT; }
};

template <class T>
class ElemList{
public:
	CkVec<T> *vec;
	ElemList(){
		vec = new CkVec<T>();
	}
	~ElemList(){
		delete vec;
	}
	ElemList(const ElemList &rhs){
		*this=rhs;
	}
	ElemList& operator=(const ElemList& rhs){
		 vec = new CkVec<T>();
		*vec = *(rhs.vec);
	}
	ElemList& operator+=(const ElemList& rhs){
		/*
			add the new unique elements to the List
		*/
		int len = vec->size();
		for(int i=0;i<rhs.vec->length();i++){
			int flag=0;
			for(int j=0;j<len;j++){
				if((*vec)[j] == (*(rhs.vec))[i]){					
					flag = 1;
					break;
				}
			}
			if(!flag){
				vec->push_back((*(rhs.vec))[i]);
			}	
		}
	}
	ElemList(const T &val){
		vec =new CkVec<T>();
		vec->push_back(val);
	};
	virtual void pup(PUP::er &p){
		if(p.isUnpacking()){
			vec = new CkVec<T>();
		}
		pupCkVec(p,*vec);
	}
};

class NodeElem {
public:
	//global number of this node
	int global;
	/*no of chunks that share this node
		owned by 1 element - numShared 0
		owned by 2 elements - numShared 2
	*/
	int numShared;
	/*somewhat horrible semantics
		-if numShared == 0 shared is NULL
		- else stores all the chunks that share this node
	*/
	int *shared;
	NodeElem(){
		global = -1;
		numShared = 0;
		shared = NULL;
	}
	NodeElem(int g_,int num_){
		global = g_; numShared= num_;
		if(numShared != 0){
			shared = new int[numShared];
		}else{
			shared = NULL;
		}
	}
	NodeElem(int g_){
		global = g_;
		numShared = 0;
		shared = NULL;
	}
	NodeElem(const NodeElem &rhs){
		shared=NULL;
		*this = rhs;
	}
	NodeElem& operator=(const NodeElem &rhs){
		global = rhs.global;
		numShared = rhs.numShared;
		if(shared != NULL){
			delete [] shared;
		}
		shared = new int[numShared];
		memcpy(shared,rhs.shared,numShared*sizeof(int));
                return *this;
	}

	bool operator == (const NodeElem &rhs){
		if(global == rhs.global){
			return true;
		}else{
			return false;
		}
	}
	virtual void pup(PUP::er &p){
		p | global;
		p | numShared;
		if(p.isUnpacking()){
			if(numShared != 0){
				shared = new int[numShared];
			}else{
				shared = NULL;
			}
		}
		p(shared,numShared);
	}
	~NodeElem(){
		if(shared != NULL){
			delete [] shared;
		}
	}
};

/*
	This class is an MSA Entity. It is used for 2 purposes
	1 It is used for storing the mesh while creating the mesh
	for each chunk
	2	It is used for storing the ghost elements and nodes for
	a chunk.
*/
class MeshElem{
	public: 
	FEM_Mesh *m;
	CkVec<int> gedgechunk; // Chunk number of 
	MeshElem(){
		m = new FEM_Mesh;
	}
	MeshElem(int dummy){
		m = new FEM_Mesh;
	}
	~MeshElem(){
		delete m;
	}
	MeshElem(const MeshElem &rhs){
		m = NULL;
		*this = rhs;
	}
	MeshElem& operator=(const MeshElem &rhs){
		if(m != NULL){
			delete m;
		}	
		m = new FEM_Mesh;
		m->copyShape(*(rhs.m));
		(*this) += rhs;
                
                return *this;
	}
	MeshElem& operator+=(const MeshElem &rhs){
		int oldel = m->nElems();
		m->copyShape(*(rhs.m));
		for(int i=0;i<rhs.m->node.size();i++){
			m->node.push_back((rhs.m)->node,i);
		}	
		if((rhs.m)->elem.size()>0){
			for(int t=0;t<(rhs.m)->elem.size();t++){
				if((rhs.m)->elem.has(t)){
					for(int e=0;e<(rhs.m)->elem.get(t).size();e++){
						m->elem[t].push_back((rhs.m)->elem.get(t),e);
					}	
				}	
			}
		}

                return *this;
	}
	virtual void pup(PUP::er &p){
		if(p.isUnpacking()){
			m = new FEM_Mesh;
		}
		m->pup(p);
	}
};


class Hashnode{
public:
	class tupledata{
		public:
		enum {MAX_TUPLE = 8};
			int nodes[MAX_TUPLE];
			tupledata(int _nodes[MAX_TUPLE]){
				memcpy(nodes,_nodes,sizeof(int)*MAX_TUPLE);
			}
			tupledata(tupledata &rhs){
				memcpy(nodes,rhs.nodes,sizeof(int)*MAX_TUPLE);
			}
			tupledata(){};
			//dont store the returned string
			char *toString(int numnodes,char *str){
				str[0]='\0';
				for(int i=0;i<numnodes;i++){
					sprintf(&str[strlen(str)],"%d ",nodes[i]);
				}
				return str;
			}
			int &operator[](int i){
				return nodes[i];
			}
			const int &operator[](int i) const {
				return nodes[i];
			}
			virtual void pup(PUP::er &p){
				p(nodes,MAX_TUPLE);
			}
	};
	int numnodes; //number of nodes in this tuple
	//TODO: replace *nodes with the above tupledata class
	tupledata nodes;	//the nodes in the tuple
	int chunk;		//the chunk number to which this element belongs 
	int elementNo;		//local number of that element
	Hashnode(){
		numnodes=0;
	};
	Hashnode(int _num,int _chunk,int _elNo,int _nodes[tupledata::MAX_TUPLE]): nodes(_nodes){
		numnodes = _num;
		chunk = _chunk;
		elementNo = _elNo;
	}
	Hashnode(const Hashnode &rhs){
		*this = rhs;
	}
	Hashnode &operator=(const Hashnode &rhs){
		numnodes = rhs.numnodes;
		for(int i=0;i<numnodes;i++){
			nodes[i] = rhs.nodes[i];
		}
		chunk = rhs.chunk;
		elementNo = rhs.elementNo;
                return *this;
	}
	bool operator==(const Hashnode &rhs){
		if(numnodes != rhs.numnodes){
			return false;
		}
		for(int i=0;i<numnodes;i++){
			if(nodes[i] != rhs.nodes[i]){
				return false;
			}
		}
		if(chunk != rhs.chunk){
			return false;
		}
		if(elementNo != rhs.elementNo){
			return false;
		}
		return true;
	}
	bool equals(tupledata &tuple){
		for(int i=0;i<numnodes;i++){
			if(tuple.nodes[i] != nodes[i]){
				return false;
			}
		}
		return true;
	}
	virtual void pup(PUP::er &p){
		p | numnodes;
		p | nodes;
		p | chunk;
		p | elementNo;
	}
};

template <class T>
ostream& operator << (ostream& os, const ElemList<T> & s){
};

template <class T>
CkOutStream& operator << (CkOutStream& os, const ElemList<T>& s) {
};

typedef MSA2D<int, DefaultEntry<int>, MSA_DEFAULT_ENTRIES_PER_PAGE, MSA_ROW_MAJOR> MSA2DRM;

typedef MSA1D<int, DefaultEntry<int>, MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1DINT;

typedef ElemList<int> IntList;
typedef MSA1D<IntList, DefaultListEntry<IntList,true>,MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1DINTLIST;

typedef ElemList<NodeElem> NodeList;
typedef MSA1D<NodeList, DefaultListEntry<NodeList,true>,MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1DNODELIST;

typedef MSA1D<MeshElem,DefaultEntry<MeshElem,true>,1> MSA1DFEMMESH;

typedef ElemList<Hashnode> Hashtuple;
typedef MSA1D<Hashtuple,DefaultListEntry<Hashtuple,true>,MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1DHASH;



struct conndata{
	int nelem;
	int nnode;
	MSA1DINT arr1;
	MSA1DINT arr2;

	void pup(PUP::er &p){
		p|nelem;
		p|nnode;
		arr1.pup(p);
		arr2.pup(p);
	}
};

/*
	Structure to store connectivity data after the 
	global element partition has been returned by parmetis
*/
struct partconndata{
	int nelem;
	int startindex;
	int *eptr,*eind;
	int *part;
	~partconndata(){
		delete [] eptr;
		delete [] eind;
		delete [] part;
	};
};

/*
	structure for storing the ghost layers
*/
struct ghostdata{
	int numLayers;
	FEM_Ghost_Layer **layers;
	void pup(PUP::er &p){
		p | numLayers;
		if(p.isUnpacking()){
			layers = new FEM_Ghost_Layer *[numLayers];
			for(int i=0;i<numLayers;i++){
				layers[i] = new FEM_Ghost_Layer;
			}

		}
		for(int i=0;i<numLayers;i++){
			layers[i]->pup(p);
		}
	}
	~ghostdata(){
			printf("destructor on ghostdata called \n");
			for(int i=0;i<numLayers;i++){
					delete layers[i];
			}
			delete [] layers;
	};
};


class MsaHashtable{
public:
	int numSlots;
	MSA1DHASH table;
	MsaHashtable(int _numSlots,int numWorkers):numSlots(_numSlots),table(_numSlots,numWorkers){
	}
	MsaHashtable(){};

	virtual void pup(PUP::er &p){
		p | numSlots;
		p | table;
	}
	int addTuple(int *tuple,int nodesPerTuple,int chunk,int elementNo){
		//sort the tuples to get a canonical form
		// bubble sort should do just as well since the number
		// of nodes is less than 10.
		for(int i=0;i<nodesPerTuple-1;i++){
			for(int j=i+1;j<nodesPerTuple;j++){
				if(tuple[j] < tuple[i]){
					int t = tuple[j];
					tuple[j] = tuple[i];
					tuple[i] = t;
				}
			}
		}
		//find out the index
		long long sum = 0;
		for(int i=0;i<nodesPerTuple;i++){
			sum = sum *numSlots + tuple[i];
		}
		int index = (int )(sum %(long )numSlots);
		Hashnode entry(nodesPerTuple,chunk,elementNo,tuple);
	
		Hashtuple &list=table.accumulate(index);
		list.vec->push_back(entry);
		char str[100];
		DEBUG(printf("[%d] adding tuple %s element %d to index %d \n",chunk,entry.nodes.toString(nodesPerTuple,str),elementNo,index));
		return index;
	}

	void print(){
		char str[100];
		for(int i=0;i<numSlots;i++){
			const Hashtuple &t = table.get(i);
			for(int j=0;j<t.vec->size();j++){
				Hashnode &tuple = (*t.vec)[j];
				printf("ghost element chunk %d element %d index %d tuple < %s>\n",tuple.chunk,tuple.elementNo,i,tuple.nodes.toString(tuple.numnodes,str));
			}
		}
	}
	void sync(){
		table.sync();
	}
	const Hashtuple &get(int i){
		return table.get(i);
	}
	
};



int FEM_master_parallel_part(int ,int ,FEM_Comm_t);
int FEM_slave_parallel_part(int ,int ,FEM_Comm_t);
struct partconndata* FEM_call_parmetis(struct conndata &data,FEM_Comm_t comm_context);
void FEM_write_nodepart(MSA1DINTLIST	&nodepart,struct partconndata *data);
void FEM_write_part2node(MSA1DINTLIST	&nodepart,MSA1DNODELIST &part2node,struct partconndata *data,MPI_Comm comm_context);
void FEM_write_part2elem(MSA1DINTLIST &part2elem,struct partconndata *data,MPI_Comm comm_context);
FEM_Mesh * FEM_break_mesh(FEM_Mesh *m,int numElements,int numChunks);
void sendBrokenMeshes(FEM_Mesh *mesh_array,FEM_Comm_t comm_context);
void	FEM_write_part2mesh(MSA1DFEMMESH &part2mesh,struct partconndata *partdata,struct conndata *data,MSA1DINTLIST &nodepart,int numChunks,int myChunk,FEM_Mesh *mypiece);
void addIDXLists(FEM_Mesh *m,NodeList &lnodes,int myChunk);
struct ghostdata *gatherGhosts();
void makeGhosts(FEM_Mesh *m,MPI_Comm comm,int masterRank,int numLayers,FEM_Ghost_Layer **layers);
void makeGhost(FEM_Mesh *m,MPI_Comm comm,int masterRank,int totalShared,FEM_Ghost_Layer *layer,	CkHashtableT<CkHashtableAdaptorT<int>,char> &sharedNode,CkHashtableT<CkHashtableAdaptorT<int>,int> &global2local);
bool sharedWith(int lnode,int chunk,FEM_Mesh *m);

// End Parallel Partitioner




/*
 Some definitions of structures/classes used in map.C and possibly other FEM source files.
 Moved to this file by Isaac Dooley 4/5/05

 FIXME: Clean up the organization of this file, and possibly move other structures from Map.C here
        Also all the stuff in this file might just belong in impl.h. I just moved it here so it could
	be included in fem.C for my element->element build adjacency table function.
*/

static FEM_Symmetries_t noSymmetries=(FEM_Symmetries_t)0;

// defined in map.C
CkHashCode CkHashFunction_ints(const void *keyData,size_t keyLen);
int CkHashCompare_ints(const void *k1,const void *k2,size_t keyLen);
extern "C" int ck_fem_map_compare_int(const void *a, const void *b);

//A linked list of elements surrounding a tuple.
//ElemLists are allocated one at a time, and hence a bit cleaner than chunklists:
class elemList {
public:
	int chunk;
	int tupleNo;//tuple number on this element for the tuple that this list sorrounds 
	int localNo;//Local number of this element on this chunk (negative for a ghost)
	int type; //Kind of element
	FEM_Symmetries_t sym; //Symmetries this element was reached via
	elemList *next;

	elemList(int chunk_,int localNo_,int type_,FEM_Symmetries_t sym_)
		:chunk(chunk_),localNo(localNo_),type(type_), sym(sym_) 
		{ next=NULL; }
	elemList(int chunk_,int localNo_,int type_,FEM_Symmetries_t sym_,int tupleNo_)
		:chunk(chunk_),localNo(localNo_),type(type_), sym(sym_) , tupleNo(tupleNo_)
		{ next=NULL; }
	
	~elemList() {if (next) delete next;}
	void setNext(elemList *n) {next=n;}
};


//Maps node tuples to element lists
class tupleTable : public CkHashtable {
  int tupleLen; //Nodes in a tuple
  CkHashtableIterator *it;
  static int roundUp(int val,int to) {
    return ((val+to-1)/to)*to;
  }
  static CkHashtableLayout makeLayout(int tupleLen) {
    int ks=tupleLen*sizeof(int);
    int oo=roundUp(ks+sizeof(char),sizeof(void *));
    int os=sizeof(elemList *);
    return CkHashtableLayout(ks,ks,oo,os,oo+os);
  }
  
	//Make a canonical version of this tuple, so different
	// orderings of the same nodes don't end up in different lists.
	//I canonicalize by sorting:
	void canonicalize(const int *tuple,int *can)
	{
		switch(tupleLen) {
		case 1: //Short lists are easy to sort:
			can[0]=tuple[0]; break;
		case 2:
			if (tuple[0]<tuple[1])
			  {can[0]=tuple[0]; can[1]=tuple[1];}
			else
			  {can[0]=tuple[1]; can[1]=tuple[0];}
			break;
		default: //Should use std::sort here:
			memcpy(can,tuple,tupleLen*sizeof(int));
			qsort(can,tupleLen,sizeof(int),ck_fem_map_compare_int);
		};
	}
public:
	enum {MAX_TUPLE=8};

	tupleTable(int tupleLen_)
		:CkHashtable(makeLayout(tupleLen_),
			     137,0.5,
			     CkHashFunction_ints,
			     CkHashCompare_ints)
	{
		tupleLen=tupleLen_;
		if (tupleLen>MAX_TUPLE) CkAbort("Cannot have that many shared nodes!\n");
		it=NULL;
	}
	~tupleTable() {
		beginLookup();
		elemList *doomed;
		while (NULL!=(doomed=(elemList *)lookupNext()))
			delete doomed;
	}
	//Lookup the elemList associated with this tuple, or return NULL
	elemList **lookupTuple(const int *tuple) {
		int can[MAX_TUPLE];
		canonicalize(tuple,can);
		return (elemList **)get(can);
	}	
	
	//Register this (new'd) element with this tuple
	void addTuple(const int *tuple,elemList *nu)
	{
		int can[MAX_TUPLE];
		canonicalize(tuple,can);
		//First try for an existing list:
		elemList **dest=(elemList **)get(can);
		if (dest!=NULL) 
		{ //A list already exists here-- link it into the new list
			nu->setNext(*dest);
		} else {//No pre-existing list-- initialize a new one.
			dest=(elemList **)put(can);
		}
		*dest=nu;
	}
	//Return all registered elemLists:
	void beginLookup(void) {
		it=iterator();
	}
	elemList *lookupNext(void) {
		void *ret=it->next();
		if (ret==NULL) {
			delete it; 
			return NULL;
		}
		return *(elemList **)ret;
	}
};




/*This object maps a single entity to a list of the chunks
that have a copy of it.  For the vast majority
of entities, the list will contain exactly one element.
*/
class chunkList : public CkNoncopyable {
public:
	int chunk;//Chunk number; or -1 if the list is empty
	int localNo;//Local number of this entity on this chunk (if negative, is a ghost)
	FEM_Symmetries_t sym; //Symmetries this entity was reached via
	int layerNo; // -1 if real; if a ghost, our ghost layer number
	chunkList *next;
	chunkList() {chunk=-1;next=NULL;}
	chunkList(int chunk_,int localNo_,FEM_Symmetries_t sym_,int ln_=-1) {
		chunk=chunk_;
		localNo=localNo_;
		sym=sym_;
		layerNo=ln_;
		next=NULL;
	}
	~chunkList() {delete next;}
	void set(int c,int l,FEM_Symmetries_t s,int ln) {
		chunk=c; localNo=l; sym=s; layerNo=ln;
	}
	//Is this chunk in the list?  If so, return false.
	// If not, add it and return true.
	bool addchunk(int c,int l,FEM_Symmetries_t s,int ln) {
		//Add chunk c to the list with local index l,
		// if it's not there already
		if (chunk==c && sym==s) return false;//Already in the list
		if (chunk==-1) {set(c,l,s,ln);return true;}
		if (next==NULL) {next=new chunkList(c,l,s,ln);return true;}
		else return next->addchunk(c,l,s,ln);
	}
	//Return this node's local number on chunk c (or -1 if none)
	int localOnChunk(int c,FEM_Symmetries_t s) const {
		const chunkList *l=onChunk(c,s);
		if (l==NULL) return -1;
		else return l->localNo;
	}
	const chunkList *onChunk(int c,FEM_Symmetries_t s) const {
		if (chunk==c && sym==s) return this;
		else if (next==NULL) return NULL;
		else return next->onChunk(c,s);
	}
	int isEmpty(void) const //Return 1 if this is an empty list 
		{return (chunk==-1);}
	int isShared(void) const //Return 1 if this is a shared entity
		{return next!=NULL;}
	int isGhost(void) const  //Return 1 if this is a ghost entity
		{return FEM_Is_ghost_index(localNo); }
	int length(void) const {
		if (next==NULL) return isEmpty()?0:1;
		else return 1+next->length();
	}
	chunkList &operator[](int i) {
		if (i==0) return *this;
		else return (*next)[i-1];
	}
};


// end of map.h header file

/* File: lock.h
 * Authors: Nilesh Choudhury
 *
 */

#define _LOCKCHUNKS

//there is one fem_lock associated with every FEM_Mesh.
class FEM_lock {
  int idx;
  int owner;
  bool isOwner;
  bool isLocked;
  bool hasLocks;
  bool isLocking;
  bool isUnlocking;
  CkVec<int> lockedChunks;
  femMeshModify *mmod;

 private:
  bool existsChunk(int index);

 public:
  FEM_lock() {};
  FEM_lock(int i, femMeshModify *m);
  ~FEM_lock();

  //locks all chunks which contain all the nodes and elements that are passed 
  //in this function
  //locking of the chunks is blocking and is strictly in ascending order.
  int lock(int numNodes, int *nodes, int numElems, int* elems, int elemType=0);
  //unlock all the concerned chunks.
  //since at one point of time one chunk can only lock one set of chunks for
  //one operation, one does not need to pass arguments to unlock.
  int unlock();
  int lock(int chunkNo, int own);
  int unlock(int chunkNo, int own);
  int getIdx() { return idx; }
};


// end lock.h


/* File: lock_node.h
 * Authors: Nilesh Choudhury
 * 
 */


//there is one fem_lock associated with every node (locks on elements are not required)
//should lock all nodes, involved in any operation
class FEM_lockN {
  int owner, pending;
  femMeshModify *theMod;
  int idx; //index of the node
  int noreadLocks;
  int nowriteLocks;
  
 public:
  FEM_lockN() {};
  FEM_lockN(int i,femMeshModify *mod);
  ~FEM_lockN();

  void reset(int i,femMeshModify *mod);
  int rlock();
  int runlock();
  int wlock(int own);
  int wunlock(int own);
  bool haslocks();
  bool verifyLock(void);
  int lockOwner();
  int getIdx() { return idx; }
};

// end lock_node.h



/* File: util.h
 * Authors: Nilesh Choudhury
 * 
 */
class FEM_MUtil {
  int idx;
  femMeshModify *mmod;

 public:
  FEM_MUtil() {}
  FEM_MUtil(int i, femMeshModify *m);
  ~FEM_MUtil();

  int getIdx() { return idx; }
  //the entType signifies what type of entity to lock. node=0, elem=1;
  //entNo signifies the local index of the entity
  //numChunks is the number of chunks that need to be locked to lock that entity
  //chunks identifies the chunks that need to be locked
  void getChunkNos(int entType, int entNo, int *numChunks, IDXL_Share ***chunks, int elemType=0);
  bool isShared(int index);
  void splitEntityAll(FEM_Mesh *m, int localIdx, int nBetween, int *between);
  void splitEntitySharing(FEM_Mesh *m, int localIdx, int nBetween, int *between, int numChunks, int *chunks);
  void splitEntityRemote(FEM_Mesh *m, int chk, int localIdx, int nBetween, int *between);
  void removeNodeAll(FEM_Mesh *m, int localIdx);
  void removeNodeRemote(FEM_Mesh *m, int chk, int sharedIdx);
  int exists_in_IDXL(FEM_Mesh *m, int localIdx, int chk, int type, int elemType=0);

  int lookup_in_IDXL(FEM_Mesh *m, int sharedIdx, int fromChk, int type, int elemType=0);
  int getRemoteIdx(FEM_Mesh *m, int elementid, int elemtype);

  void addGhostElementRemote(FEM_Mesh *m, int chk, int elemType, int numGhostIndices, int *ghostIndices, int numSharedIndices, int *sharedIndices, int connSize);
  chunkListMsg *getChunksSharingGhostNodeRemote(FEM_Mesh *m, int chk, int sharedIdx);
  void buildChunkToNodeTable(int *nodetype, int sharedcount, int ghostcount, int localcount, int *conn, int connSize, CkVec<int> ***allShared, int *numSharedChunks, CkVec<int> **allChunks, int ***sharedConn);
  void addElemRemote(FEM_Mesh *m, int chk, int elemtype, int connSize, int *conn, int numGhostIndex, int *ghostIndices);
  void removeGhostElementRemote(FEM_Mesh *m, int chk, int elementid, int elemtype, int numGhostIndex, int *ghostIndices, int numGhostRNIndex, int *ghostRNIndices, int numGhostREIndex, int *ghostREIndices, int numSharedIndex, int *sharedIndices);
  void removeElemRemote(FEM_Mesh *m, int chk, int elementid, int elemtype, int permanent);
  void removeGhostNodeRemote(FEM_Mesh *m, int fromChk, int sharedIdx);
  int Replace_node_local(FEM_Mesh *m, int oldIdx, int newIdx);
  void addToSharedList(FEM_Mesh *m, int fromChk, int sharedIdx);
  int eatIntoElement(int localIdx);
  int getLockOwner(int nodeId);
  bool knowsAbtNode(int chk, int nodeId);
  void UpdateGhostSend(int nodeId, int *chunkl, int numchunkl);
  void findGhostSend(int nodeId, int **chunkl, int *numchunkl);

  void StructureTest(FEM_Mesh *m);
  int AreaTest(FEM_Mesh *m);
  int IdxlListTest(FEM_Mesh *m);
  void verifyIdxlListRemote(FEM_Mesh *m, int fromChk, int fsize, int type);
  int residualLockTest(FEM_Mesh *m);

  void FEM_Print_n2n(FEM_Mesh *m, int nodeid);
  void FEM_Print_n2e(FEM_Mesh *m, int nodeid);
  void FEM_Print_e2n(FEM_Mesh *m, int eid);
  void FEM_Print_e2e(FEM_Mesh *m, int eid);
  void FEM_Print_coords(FEM_Mesh *m, int nodeid);

  void idxllock(FEM_Mesh *m, int chk, int type);
  void idxlunlock(FEM_Mesh *m, int chk, int type);
  void idxllockLocal(FEM_Mesh *m, int toChk, int type);
  void idxlunlockLocal(FEM_Mesh *m, int toChk, int type);

  void copyElemData(int etype, int elemid, int newEl);
};



// End util.h
/* File: ParFUM_adapt_lock.h
 * Authors: Nilesh Choudhury, Terry Wilmarth
 *
 */




/* File: fem_mesh_modify.h
 * Authors: Nilesh Choudhury
 * 
 */

/*
 
This file contains a set of functions, which allow primitive operations upon meshes in parallel. The functions are defined in fem_mesh_modify.C.


Assumptions:

The mesh must be in a consistant state before and after these operations:
- Any shared node must be in the IDXL table for both the local and 
other chunks.
- FEM_add_element can only be called with a set of existing local or shared nodes
- The mesh must be nice. Each element may be adjacent to at most one other element per face/edge/tuple
- The mesh must have e2e, n2n, and n2e adjacencies computed before any of these functions are called.
The calls will maintain the adjacency tables, both remotely and locally.
- Exactly one ghost layer exists around all chunks before these modification functions are called. 
A ghost element is one that is adjacent to at least one shared node. A ghost node is any non-shared 
node adjacent to a ghost element. The e2e adjacencies need not have the same definition for 
adjacent elements.

*/

#define MAX_CHUNK 1000000000


// The internal functions which take in a FEM_Mesh*, but could feasibly be used by others
int FEM_add_node(FEM_Mesh *m, int* adjacent_nodes=0, int num_adjacent_nodes=0, int *chunks=0, int numChunks=0, int forceShared=0, int upcall=0);
void FEM_remove_node(FEM_Mesh *m, int node);
int FEM_remove_element(FEM_Mesh *m, int element, int elem_type=0, int permanent=-1);
int FEM_purge_element(FEM_Mesh *m, int element, int elem_type=0);
int FEM_add_element(FEM_Mesh *m, int* conn, int conn_size, int elem_type=0, int chunkNo=-1);
int FEM_Modify_Lock(FEM_Mesh *m, int* affectedNodes=0, int numAffectedNodes=0, int* affectedElts=0, int numAffectedElts=0, int elemtype=0);
int FEM_Modify_Unlock(FEM_Mesh *m);
int FEM_Modify_LockN(FEM_Mesh *m, int nodeId, int readLock);
int FEM_Modify_UnlockN(FEM_Mesh *m, int nodeId, int readLock);
void FEM_Modify_LockAll(FEM_Mesh*m, int nodeId, bool lockall=true);
void FEM_Modify_LockUpdate(FEM_Mesh*m, int nodeId, bool lockall=true);
void FEM_Modify_correctLockN(FEM_Mesh *m, int nodeId);

// Internal functions which shouldn't be used by anyone else
int FEM_add_node_local(FEM_Mesh *m, int addGhost=0);
void FEM_remove_node_local(FEM_Mesh *m, int node);
int FEM_add_element_local(FEM_Mesh *m, const int *conn, int connSize, int elemType, int addGhost);
void FEM_remove_element_local(FEM_Mesh *m, int element, int etype);

void FEM_Ghost_Essential_attributes(FEM_Mesh *m, int coord_attr, int bc_attr, int nodeid);

void FEM_Mesh_dataP(FEM_Mesh *fem_mesh,int entity,int attr,void *data, int firstItem, int length, int datatype,int width);
void FEM_Mesh_data_layoutP(FEM_Mesh *fem_mesh,int entity,int attr,void *data, int firstItem, int length, IDXL_Layout_t layout);
void FEM_Mesh_data_layoutP(FEM_Mesh *fem_mesh,int entity,int attr,void *data, int firstItem,int length, const IDXL_Layout &layout);


class femMeshModMsg : public CMessage_femMeshModMsg {
 public:
  int numChunks;
  int myChunk;

  femMeshModMsg() {}
  
  femMeshModMsg(int num, int idx) {
    numChunks = num;
    myChunk = idx;
  }
  
  ~femMeshModMsg() {}
};

class boolMsg : public CMessage_boolMsg {
 public:
  bool b;

  boolMsg(bool bo) {
    b = bo;
  }

  ~boolMsg() {}
};

class intMsg : public CMessage_intMsg {
 public:
  int i;

  intMsg(int n) {
    i = n;
  }

  ~intMsg(){}
};

class int2Msg : public CMessage_int2Msg {
 public:
  int i, j;

  int2Msg(int m, int n) {
    i = m;
    j = n;
  }

  ~int2Msg(){}
};

class double2Msg : public CMessage_double2Msg {
 public:
  double i,j;

  double2Msg(double m, double n) {
    i = m;
    j = n;
  }

  ~double2Msg() {}
};

class FEMMeshMsg : public CMessage_FEMMeshMsg {
 public:
  FEM_Mesh *m;
  int dimn;

  FEMMeshMsg(FEM_Mesh *mh, int dim) {
    m = mh;
    dimn = dim;
  }

  ~FEMMeshMsg() {}
};

class addNodeMsg : public CMessage_addNodeMsg {
 public:
  int chk;
  int nBetween;
  int *between;
  int *chunks;
  int numChunks;
  int forceShared;
  int upcall;

  ~addNodeMsg() {
    if(between) {
      //delete between;
      //delete chunks;
    }
  }
};

class sharedNodeMsg : public CMessage_sharedNodeMsg {
 public:
  int chk;
  int nBetween;
  int *between;

  /*sharedNodeMsg(int c, int nB, int *B) {
    chk = c;
    nBetween = nB;
    between = (int *)malloc(nBetween*sizeof(int));
    for(int i=0; i<nBetween; i++) {
    between[i] = B[i];
    }
    }*/

  ~sharedNodeMsg() {
    //if(between) {
    //delete between;
    //}
  }
};

class removeSharedNodeMsg : public CMessage_removeSharedNodeMsg {
 public:
  int chk;
  int index;
};

class addGhostElemMsg : public CMessage_addGhostElemMsg {
 public:
  int chk;
  int elemType;
  int numGhostIndex;
  int *ghostIndices;
  int numSharedIndex;
  int *sharedIndices;
  int connSize;

  ~addGhostElemMsg() {
    if(ghostIndices) {
      //delete ghostIndices;
    }
    if(sharedIndices) {
      //delete sharedIndices;
    }
  }
};

class chunkListMsg : public CMessage_chunkListMsg {
 public:
  int numChunkList;
  int *chunkList;
  int *indexList;

  ~chunkListMsg() {
    if(numChunkList>0) {
      //delete chunkList;
      //delete indexList;
    }
  }
};

class addElemMsg : public CMessage_addElemMsg {
 public:
  int chk;
  int elemtype;
  int connSize;
  int *conn;
  int numGhostIndex;
  int *ghostIndices;

  ~addElemMsg() {
    if(conn) {
      //delete conn;
    }
    if(ghostIndices) {
      //delete ghostIndices;
    }
  }
};

class removeGhostElemMsg : public CMessage_removeGhostElemMsg {
 public:
  int chk;
  int elemtype;
  int elementid;
  int numGhostIndex;
  int numGhostRNIndex;
  int numGhostREIndex;
  int numSharedIndex;
  int *ghostIndices;
  int *ghostRNIndices;
  int *ghostREIndices;
  int *sharedIndices;

  ~removeGhostElemMsg() {
    if(ghostIndices) {
      //   delete ghostIndices;
      //   delete ghostRNIndices;
      //    delete ghostREIndices;
      //    delete sharedIndices;
    }
  }
};

class removeElemMsg : public CMessage_removeElemMsg {
 public:
  int chk;
  int elementid;
  int elemtype;
  int permanent;
};

class verifyghostsendMsg : public CMessage_verifyghostsendMsg {
 public:
  int fromChk;
  int sharedIdx;
  int numchks;
  int *chunks;
  
  ~verifyghostsendMsg() {
  }
};

class findgsMsg : public CMessage_findgsMsg {
 public:
  int numchks;
  int *chunks;
  
  ~findgsMsg() {
  }
};

class elemDataMsg : public CMessage_elemDataMsg {
 public:
  int datasize;
  char *data;

  elemDataMsg(int size) {
    datasize = size;
  }
};

class FEM_Interpolate;

class femMeshModify : public CBase_femMeshModify {
  friend class FEM_lock;
  friend class FEM_MUtil;
  friend class FEM_Mesh;
  friend class FEM_Interpolate;
  friend class FEM_Adapt;
  friend class FEM_AdaptL;
  friend class FEM_Adapt_Algs;

 public:
  int numChunks;
  int idx;
  FEM_Mesh *fmMesh;
  FEM_lock *fmLock;
  CkVec<FEM_lockN *> fmLockN;
  //CkVec<FEM_lockN *> *fmgLockN;
  CkVec<bool> fmIdxlLock; //each chunk can have numChunks*5 idxl lists. 
  CkVec<int> fmfixedNodes; //this list is populated initially, and never changes (defines shape)
  FEM_MUtil *fmUtil;
  FEM_Interpolate *fmInp;
  FEM_Adapt *fmAdapt;
  FEM_AdaptL *fmAdaptL;
  FEM_Adapt_Algs *fmAdaptAlgs;

 public:
  femMeshModify(femMeshModMsg *fm);
  femMeshModify(CkMigrateMessage *m)/* : TCharmClient1D(m) */{};
  ~femMeshModify();

  intMsg *lockRemoteChunk(int2Msg *i2msg);
  intMsg *unlockRemoteChunk(int2Msg *i2msg);
  intMsg *lockRemoteNode(int sharedIdx, int fromChk, int isGhost, int readLock);
  intMsg *unlockRemoteNode(int sharedIdx, int fromChk, int isGhost, int readLock);
  void setFemMesh(FEMMeshMsg *fm);
  int getNumChunks(){return numChunks;}
  int getIdx(){return idx;}
  FEM_Mesh *getfmMesh(){return fmMesh;}
  FEM_lock *getfmLock(){return fmLock;}
  FEM_lockN *getfmLockN(int nodeid){
    /*if(!FEM_Is_ghost_index(nodeid)) {
      return fmLockN[nodeid];
      } else {
      return fmgLockN[FEM_To_ghost_index(nodeid)];
      }*/
    CkAssert(nodeid < fmLockN.size());
    return fmLockN[nodeid];
  }
  FEM_MUtil *getfmUtil(){return fmUtil;}
  FEM_Adapt *getfmAdapt(){return fmAdapt;}
  FEM_AdaptL *getfmAdaptL(){return fmAdaptL;}
  FEM_Adapt_Algs *getfmAdaptAlgs(){return fmAdaptAlgs;}
  FEM_Interpolate *getfmInp(){return fmInp;}

  intMsg *addNodeRemote(addNodeMsg *fm);
  void addSharedNodeRemote(sharedNodeMsg *fm);
  void removeSharedNodeRemote(removeSharedNodeMsg *fm);

  void addGhostElem(addGhostElemMsg *fm);
  chunkListMsg *getChunksSharingGhostNode(int2Msg *);
  void addElementRemote(addElemMsg *fm);

  void removeGhostElem(removeGhostElemMsg *fm);
  void removeElementRemote(removeElemMsg *fm);

  void removeGhostNode(int fromChk, int sharedIdx);

  intMsg *eatIntoElement(int fromChk, int sharedIdx);
  intMsg *getLockOwner(int fromChk, int sharedIdx);
  boolMsg *knowsAbtNode(int fromChk, int toChk, int sharedIdx);

  void refine_flip_element_leb(int fromChk, int propElemT, int propNodeT,
			       int newNodeT, int nbrOpNodeT, int nbrghost,
			       double longEdgeLen);

  void addToSharedList(int fromChk, int sharedIdx);
  void updateNodeAttrs(int fromChk, int sharedIdx, double coordX, double coordY, int bound, bool isGhost);
  void updateghostsend(verifyghostsendMsg *vmsg);
  findgsMsg *findghostsend(int fromChk, int sharedIdx);

  double2Msg *getRemoteCoord(int fromChk, int ghostIdx);
  intMsg *getRemoteBound(int fromChk, int ghostIdx);

  intMsg *getIdxGhostSend(int fromChk, int idxshared, int toChk);
  void updateIdxlList(int fromChk, int idxTrans, int transChk);
  void removeIDXLRemote(int fromChk, int sharedIdx, int type);
  void addTransIDXLRemote(int fromChk, int sharedIdx, int type);
  void verifyIdxlList(int fromChk, int size, int type);

  void idxllockRemote(int fromChk, int type);
  void idxlunlockRemote(int fromChk, int type);

  intMsg *hasLockRemoteNode(int sharedIdx, int fromChk, int isGhost);
  void modifyLockAll(int fromChk, int sharedIdx);
  boolMsg *verifyLock(int fromChk, int sharedIdx, int isGhost);
  void verifyghostsend(verifyghostsendMsg *vmsg);
  boolMsg *shouldLoseGhost(int fromChk, int sharedIdx, int toChk);

  void addghostsendl(int fromChk, int sharedIdx, int toChk, int transIdx);
  void addghostsendl1(int fromChk, int transChk, int transIdx);
  void addghostsendr(int fromChk, int sharedIdx, int toChk, int transIdx);
  void addghostsendr1(int fromChk, int transChk, int transIdx);
  boolMsg *willItLose(int fromChk, int sharedIdx);

  void interpolateElemCopy(int fromChk, int sharedIdx1, int sharedIdx2);
  void cleanupIDXL(int fromChk, int sharedIdx);
  void purgeElement(int fromChk, int sharedIdx);
  elemDataMsg *packElemData(int fromChk, int sharedIdx);
};
// end mesh_modify.h





/* File: adapt_algs.h
 * Authors: Terry Wilmarth, Nilesh Choudhury
 * 
 */

// This module implements high level mesh adaptivity algorithms that make use 
// of the primitive mesh adaptivity operations provided by fem_adapt(_new).
// Ask: TLW

#define SLIVERAREA 1.0e-18
#define REFINE_TOL 1.3  // Refine elements with average edge length > 
                        // REFINE_TOL*desiredEdgeLength
#define COARSEN_TOL 0.6 // Coarsen element with average edge length <
                        // COARSEN_TOL*desiredEdgeLength
#define QUALITY_MIN 0.3

class FEM_Adapt_Algs;
CtvExtern(FEM_Adapt_Algs *, _adaptAlgs);

class femMeshModify;
class FEM_Adapt;
class FEM_AdaptL;

class FEM_Adapt_Algs {
  friend class FEM_AdaptL;
  friend class FEM_Adapt;
  friend class femMeshModify;
  friend class FEM_Interpolate;
  friend class FEM_MUtil;

 public:
  int coord_attr;
  int bc_attr;

 protected: 
  FEM_Mesh *theMesh;
  femMeshModify *theMod;
  //FEM_Adapt *theAdaptor;
  FEM_AdaptL *theAdaptor;
  int numNodes, numElements, dim;
  // These are for element sorting
  typedef struct {
    int elID;
    double len;
  } elemHeap;
  elemHeap *coarsenElements;
  elemHeap *refineElements;
  elemHeap *refineStack;
  int refineTop, refineHeapSize, coarsenHeapSize;

 public:
  FEM_Adapt_Algs() {
    theMesh = NULL; theMod = NULL; theAdaptor = NULL;
  }
  /// Initialize FEM_Adapt_Algs with a chunk of the mesh
  FEM_Adapt_Algs(FEM_Mesh *m, femMeshModify *fm, int dimension);
  void FEM_Adapt_Algs_Init(int coord_at, int bc_at) {
    coord_attr = coord_at;
    bc_attr = bc_at;
  }
  /// Perform refinements on a mesh
  /** Perform refinements on a mesh.  Tries to maintain/improve element quality
      as specified by a quality measure qm;
      if method = 0, refine areas with size larger than factor down to factor
      if method = 1, refine elements down to sizes specified in sizes array
      Negative entries in size array indicate no refinement. */
  void FEM_Refine(int qm, int method, double factor, double *sizes);
  /// Perform coarsening on a mesh
  /** Perform coarsening on a mesh.  Tries to maintain/improve element quality
      as specified by a quality measure qm;
      if method = 0, coarsen areas with size smaller than factor up to factor
      if method = 1, coarsen elements up to sizes specified in sizes array
      Negative entries in size array indicate no coarsening. */
  void FEM_Coarsen(int qm, int method, double factor, double *sizes);
  /// Perform refinement/coarsening on a mesh
    /** Same as above */
  void FEM_AdaptMesh(int qm, int method, double factor, double *sizes);
  /// Smooth the mesh using method according to some quality measure qm
  void FEM_Smooth(int qm, int method);
  /// Repair the mesh according to some quality measure qm

  // FEM_Mesh_mooth
  //	Inputs	: meshP - a pointer to the FEM_Mesh object to smooth
  //		: nodes - an array of local node numbers to be smoothed.  Send
  //			  NULL pointer to smooth all nodes.
  //		: nNodes - the size of the nodes array
  //		: attrNo - the attribute number where the coords are registered
  //	Shifts nodes around to improve mesh quality.  FEM_BOUNDARY attribute
  //	and interpolator function must be registered by user to maintain 
  //	boundary information.
  void FEM_mesh_smooth(FEM_Mesh *meshP, int *nodes, int nNodes, int attrNo);

  void FEM_Repair(int qm);
  /// Remesh entire mesh
  /** Remesh entire mesh according to quality measure qm
      if method = 0, set entire mesh size to factor
      if method = 1, keep regional mesh sizes, and scale by factor
      if method = 2, uses sizes to size mesh by regions */
  void FEM_Remesh(int qm, int method, double factor, double *sizes);
  
  /// Set sizes on mesh elements based on their average edge length
  void SetReferenceMesh();
  /// Adjust sizes on mesh elements to avoid sharp discontinuities
  void GradateMesh(double smoothness);
 private:
  // Helper methods
  /// Performs refinement; returns number of modifications
  int Refine(int qm, int method, double factor, double *sizes);
  /// Performs coarsening; returns number of modifications
  int Coarsen(int qm, int method, double factor, double *sizes);
  /// Set sizes on elements throughout the mesh; note: size is edge length
  void SetMeshSize(int method, double factor, double *sizes);
  /// Insert element to be refined/coarsened
  void Insert(int elID, double len, int cFlag);
  /// Get next element to be refined/coarsened
  int Delete_Min(int cFlag);
 public:
  /// Initiate instance of Longest Edge Bisection on an element
  /** Initiate instance of Longest Edge Bisection on element e.  Propagates
      throughout the mesh to maintain the requirement that only longest edges
      are bisected; returns 1 if successful, 0 if not **/
  virtual int refine_element_leb(int e);
  virtual void refine_flip_element_leb(int e, int p, int n1, int n2, 
				       double le);

  int simple_refine(double targetA, double xmin=0.0, double ymin=0.0, double xmax=1.0, double ymax=1.0);
  int simple_coarsen(double targetA, double xmin=0.0, double ymin=0.0, double xmax=1.0, double ymax=1.0);
  double length(int n1, int n2);
  double getArea(int n1, int n2, int n3);
  double length(double *n1_coord, double *n2_coord);
  double getArea(double *n1_coord, double *n2_coord, double *n3_coord);
  int getCoord(int n1, double *crds);
  int getShortestEdge(int n1, int n2, int n3, int* shortestEdge);
  double getAreaQuality(int elem);
  bool didItFlip(int n1, int n2, int n3, double *n4_coord);
  bool didItFlip(double *n1_coord, double *n2_coord, double *n3_coord, double *n4_coord);
  double getSignedArea(double *n1_coord, double *n2_coord, double *n3_coord);
  void tests(void);
};



// End Adapt Algs





/* File: fem_adapt_new.h
 * Authors: Terry Wilmarth, Nilesh Choudhury
 *
 */

class femMeshModify;

class FEM_Adapt {
 protected:
  FEM_Mesh *theMesh;
  femMeshModify *theMod;
  // Helper methods: see bottom of this file
  /// Check if e1 and e3 are on the same side of edge path (n1, n, n2)
  /** Makes use of ordering of nodes in e1 to check is e3 is on the same side
      of the path of edges (n1, n) and (n, n2) **/
  int check_orientation(int e1, int e3, int n, int n1, int n2);
 public:
  /// Map a pair of element-local node numberings to an element-local edge 
  /// numbering
  /** Given two element-local node numberings (i.e. 0, 1, 2 for triangular 
      elements), calculate an element-local edge numbering (also 0, 1, or 2
      for triangular elements) **/
  int get_edge_index(int local_node1, int local_node2);
  /// Find an element-local node numbering for a chunk-local node
  /** Given a chunk-local element number e and a chunk-local node number n,
      determine the element-local node numbering for node n on element e **/
  int find_local_node_index(int e, int n);
  /// Extract adjacency data relative to edge [n1,n2]
  /** Extract elements adjacent to edge [n1,n2] along with element-local node
      numberings and nodes opposite input edge **/
  int findAdjData(int n1, int n2, int *e1, int *e2, int *e1n1, int *e1n2, 
		   int *e1n3, int *e2n1, int *e2n2, int *e2n3, int *n3, 
		   int *n4);
  int e2n_getNot(int e, int n1, int n2) {
    int eConn[3];
    theMesh->e2n_getAll(e, eConn);
    for (int i=0; i<3; i++) 
      if ((eConn[i] != n2) && (eConn[i] != n1)) return eConn[i];
    return -1; //should never come here
  }
  int n2e_exists(int n, int e) {
    int *nConn, nSz;
    theMesh->n2e_getAll(n, &nConn, &nSz);
    for (int i=0; i<nSz; i++) {
      if (nConn[i] == e) {
	if(nSz!=0) free(nConn);
	return 1;
      }
    }
    if(nSz!=0) free(nConn);
    return 0;
  }
  int findElementWithNodes(int n1, int n2, int n3) {
    int *nConn, nSz;
    int ret = -1;
    theMesh->n2e_getAll(n1, &nConn, &nSz);
    for (int i=0; i<nSz; i++) {
      if ((n2e_exists(n2, nConn[i])) && (n2e_exists(n3, nConn[i]))) {
	ret = nConn[i];
	break;
      }
    }
    if(nSz!=0) free(nConn);
    return ret; //should never come here
  }
  int getSharedNodeIdxl(int n, int chk);
  int getGhostNodeIdxl(int n, int chk);
  int getGhostElementIdxl(int e, int chk);

  FEM_Adapt() {
    theMesh = NULL; theMod = NULL;
  }

  /// Initialize FEM_Adapt with a chunk of the mesh
  FEM_Adapt(FEM_Mesh *m, femMeshModify *fm) { theMesh = m; theMod = fm; }
  /// Perform a Delaunay flip of edge (n1, n2)
  /** Perform a Delaunay flip of edge (n1, n2) returning 1 if successful, 0 if 
      not (likely due to the edge being on a boundary).  The convexity of the 
      quadrilateral formed by two faces incident to edge (n1, n2) is assumed. 
      n1 and n2 are assumed to be local to this chunk. An adjacency test is
      performed on n1 and n2 by searching for an element with edge [n1,n2]. **/
  //virtual int edge_flip(int n1, int n2);
  int edge_flip_help(int e1, int e2, int n1, int n2, int e1_n1, 
			     int e1_n2, int e1_n3, int n3, int n4,int *locknodes);
  
  /// Bisect edge (n1, n2) and the two adjacent elements
  /** Given edge e:(n1, n2), remove the two elements (n1,n2,n3) and (n2,n1,n4) 
      adjacent to e, and bisect e by adding node n5. Add elements (n1,n5,n3), 
      (n5,n2,n3), (n5,n1,n4) and (n2,n5,n4); returns new node n5. **/
  //virtual int edge_bisect(int n1, int n2);
  int edge_bisect_help(int e1, int e2, int n1, int n2, int e1_n1, 
			       int e1_n2, int e1_n3, int e2_n1, int e2_n2, 
			       int e2_n3, int n3, int n4);

  /// Remove the degree 4 vertex n1 without modifying degree of adj n2
  /** Inverse of edge bisect, this removes a degree 4 vertex n1 and 2 of its
      adjacent elements.  n2 indicates that the two elements removed are
      adjacent to edge [n1,n2]. This could be performed with edge_contraction,
      but this is a simpler operation. **/
  //virtual int vertex_remove(int n1, int n2);
  int vertex_remove_help(int e1, int e2, int n1, int n2, int e1_n1, 
				 int e1_n2, int e1_n3, int e2_n1, int e2_n2, 
				 int e2_n3, int n3, int n4, int n5);

  /// Contract edge (n1, n2) and the two adjacent elements
  /** Given and edge e:(n1, n2), determine the two adjacent elements (n1,n2,n3)
      and (n1,n2,n4). Contract edge e by creating node n5, removing all 
      elements incident on n1 xor n2 and reinserting with incidence on n5, 
      removing the two elements (n1,n2,n3) and (n1,n2,n4) adjacent to e, and 
      finally removing nodes n1 and n2; return 1 if successful, 0 if not **/
  //virtual int edge_contraction(int n1, int n2);
  //virtual int edge_contraction_help(int *e1P, int *e2P, int n1, int n2, int e1_n1, int e1_n2, int e1_n3, int e2_n1, int e2_n2, int e2_n3, int n3, int n4);

  /// Split a node n into two nodes with an edge in between
  /** Given a node n and two adjacent nodes n1 and n2, split n into two nodes
      n and np such that the edges to the neighbors n1 and n2
      expand into two new elements (n, np, n1) and (np, n, n2);
      return the id of the newly created node np **/
  virtual int vertex_split(int n, int n1, int n2);
  int vertex_split_help(int n, int n1, int n2, int e1, int e3);

  virtual void printAdjacencies(int *nodes, int numNodes, int *elems, int numElems);

  virtual bool isFixedNode(int n1);
  virtual bool isCorner(int n1);
  virtual bool isEdgeBoundary(int n1, int n2);
};



// End adapt_new.h



class FEM_AdaptL : public FEM_Adapt {
 public:
  FEM_AdaptL() {
    theMesh = NULL; theMod = NULL;
  }
  
  /// Initialize FEM_Adapt with a chunk of the mesh
  FEM_AdaptL(FEM_Mesh *m, femMeshModify *fm) { theMesh = m; theMod = fm; }

  int lockNodes(int *, int *, int, int *, int);
  int unlockNodes(int *, int *, int, int *, int);
  int edge_flip(int n1, int n2);
  int edge_bisect(int n1, int n2);
  int vertex_remove(int n1, int n2);
  int edge_contraction(int n1, int n2);
  int edge_contraction_help(int *e1P, int *e2P, int n1, int n2, int e1_n1, 
				    int e1_n2, int e1_n3, int e2_n1, int e2_n2,
				    int e2_n3, int n3, int n4);
};

// End Adapt_lock.h



/* File: interpolate.h
 * Authors: Terry Wilmarth, Nilesh Choudhury
 * 
 */

// This is an interface for specifying solution transfer operations
// for small mesh modification operations. It provides overridable
// defaults for node and element interpolations.  These functions are passed to
// FEM_add_element and FEM_add_node operations, along with a simple argument
// struct - TLW

class femMeshModify;

class FEM_Interpolate {
  FEM_Mesh *theMesh;
  femMeshModify *theMod;
 public:
  typedef struct {
    int n;
    int nodes[8];
    int dim;
    int nNbrs;
    double coord[3];
    double frac;
    bool addNode;
  } NodalArgs;

  typedef struct {
    int e;
    int oldElement;
    int elType;
  } ElementArgs;

  typedef void (* FEM_InterpolateNodeFn)(NodalArgs, FEM_Mesh *);
  typedef void (* FEM_InterpolateElementFn)(ElementArgs);

  FEM_InterpolateNodeFn nodeEdgeFnPtr, nodeFaceFnPtr, nodeElementFnPtr, nodeCopyFnPtr;
  FEM_InterpolateElementFn elemCopyFnPtr, elemNodeFnPtr;

  /// Basic Constructor
  FEM_Interpolate() {
    nodeEdgeFnPtr = nodeFaceFnPtr = nodeElementFnPtr = nodeCopyFnPtr = NULL;
    elemCopyFnPtr = elemNodeFnPtr = NULL;
  }

  FEM_Interpolate(FEM_Mesh *m) {
    nodeEdgeFnPtr = nodeFaceFnPtr = nodeElementFnPtr = nodeCopyFnPtr = NULL;
    elemCopyFnPtr = elemNodeFnPtr = NULL;
    theMesh = m;
  }

  FEM_Interpolate(FEM_Mesh *m, femMeshModify *fm) {
    nodeEdgeFnPtr = nodeFaceFnPtr = nodeElementFnPtr = nodeCopyFnPtr = NULL;
    elemCopyFnPtr = elemNodeFnPtr = NULL;
    theMesh = m;
    theMod = fm;
  }

  void FEM_InterpolateSetMesh(FEM_Mesh *m) { theMesh = m; }
  
  // Methods to set and reset interpolate functions on the fly; these will be
  // used to override defaults by Fortan code, and can also be set and reset
  // for using temporary special-purpose interpolation functions
  void FEM_SetInterpolateNodeEdgeFnPtr(FEM_InterpolateNodeFn fnPtr) {
    nodeEdgeFnPtr = fnPtr;
  }
  void FEM_SetInterpolateNodeFaceFnPtr(FEM_InterpolateNodeFn fnPtr) {
    nodeFaceFnPtr = fnPtr;
  }
  void FEM_SetInterpolateNodeElementFnPtr(FEM_InterpolateNodeFn fnPtr) {
    nodeElementFnPtr = fnPtr;
  }
  void FEM_SetInterpolateElementCopyFnPtr(FEM_InterpolateElementFn fnPtr) {
    elemCopyFnPtr = fnPtr;
  }
  void FEM_SetInterpolateElementNodeFnPtr(FEM_InterpolateElementFn fnPtr) {
    elemNodeFnPtr = fnPtr;
  }
  void FEM_SetInterpolateCopyAttributesFnPtr(FEM_InterpolateNodeFn fnPtr) {
    nodeCopyFnPtr = fnPtr;
  }
  void FEM_ResetInterpolateNodeEdgeFnPtr() { nodeEdgeFnPtr = NULL; }
  void FEM_ResetInterpolateNodeFaceFnPtr() { nodeFaceFnPtr = NULL; }
  void FEM_ResetInterpolateNodeElementFnPtr() { nodeElementFnPtr = NULL; }
  void FEM_ResetInterpolateElementCopyFnPtr() { elemCopyFnPtr = NULL; }  
  void FEM_ResetInterpolateElementNodeFnPtr() { elemNodeFnPtr = NULL; }
  void FEM_ReetInterpolateCopyAttributesFnPtr() { nodeCopyFnPtr = NULL; }

  // Nodal data
  /// A node is added on an edge; interpolate from neighboring nodes
  /** A node is added on an edge; interpolate from neighboring nodes; this
      uses n, nodes[2], dim and frac. Frac is between 0.0 and 1.0 and weights 
      nodes[0]; i.e. if frac=1.0, n gets a copy of nodes[0]'s data, and
      nodes[0]'s coords **/
  virtual void FEM_InterpolateNodeOnEdge(NodalArgs args);
  /// A node is added on an face; interpolate from nodes of face
  /** A node is added on an face; interpolate from nodes of face; this uses
      n, nodes[3] or nodes[4] depending on element type, dim and coord **/
  virtual void FEM_InterpolateNodeOnFace(NodalArgs args);
  /// A node is added inside a volume; interpolate from nodes of element
  /** A node is added inside a volume; interpolate from nodes of element; this
      uses n, nodes[4] or more, and coord **/
  virtual void FEM_InterpolateNodeInElement(NodalArgs args);

  // Element data
  /// An element added is completely encapsulated by element to be removed
  /** An element added is completely encapsulated by element to be removed;
      For example, edge bisect two elements replace the old element, and we
      simply copy the data of the old element to both new elements **/
  virtual void FEM_InterpolateElementCopy(ElementArgs args);
  /// An element is added and derives data from its nodes
  /** An element is added and derives data from its nodes; assumes relevant 
      data was copied to the appropriate nodes prior to this operation; see
      the utility function below **/
  virtual void FEM_InterpolateElementFromNodes(ElementArgs args);

  // Utility
  /// Store data of an element temporarily on all nodes
  /** Store data of an element temporarily on all nodes; this data is used 
      later to derive an element's data **/
  virtual void FEM_InterpolateElementToNodes(int e);

  //node data
  virtual void FEM_InterpolateCopyAttributes(int oldnode, int newnode);
};


// End interpolate.h



/*
Orion's Standard Library
Orion Sky Lawlor, 2/22/2000
NAME:		vector2d.h

DESCRIPTION:	C++ 2-Dimentional vector library (no templates)

This file provides various utility routines for easily
manipulating 2-D vectors-- included are arithmetic,
dot product, magnitude and normalization terms. 
All routines are provided right in the header file (for inlining).

Converted from vector3d.h.

*/

#ifndef __OSL_VECTOR_2D_H
#define __OSL_VECTOR_2D_H

#include <math.h>

typedef double real;

//vector2d is a cartesian vector in 2-space-- an x and y.
class vector2d {
public:
	real x,y;
	vector2d(void) {}//Default consructor
	//Simple 1-value constructor
	explicit vector2d(const real init) {x=y=init;}
	//Simple 1-value constructor
	explicit vector2d(int init) {x=y=init;}
	//2-value constructor
	vector2d(const real Nx,const real Ny) {x=Nx;y=Ny;}
	//Copy constructor
	vector2d(const vector2d &copy) {x=copy.x;y=copy.y;}
	
	//Cast-to-real * operators (treat vector as array)
	operator real *() {return &x;}
	operator const real *() const {return &x;}
	
/*Arithmetic operations: these are carefully restricted to just those
 that make unambiguous sense (to me... now...  ;-)
Counterexamples: vector*vector makes no sense (use .dot()) because
real/vector is meaningless (and we'd want a*b/b==a for b!=0), 
ditto for vector&vector (dot?), vector|vector (projection?), 
vector^vector (cross?),real+vector, vector+=real, etc.
*/
	vector2d &operator=(const vector2d &b) {x=b.x;y=b.y;return *this;}
	int operator==(const vector2d &b) const {return (x==b.x)&&(y==b.y);}
	int operator!=(const vector2d &b) const {return (x!=b.x)||(y!=b.y);}
	vector2d operator+(const vector2d &b) const {return vector2d(x+b.x,y+b.y);}
	vector2d operator-(const vector2d &b) const {return vector2d(x-b.x,y-b.y);}
	vector2d operator*(const real scale) const 
		{return vector2d(x*scale,y*scale);}
	friend vector2d operator*(const real scale,const vector2d &v)
		{return vector2d(v.x*scale,v.y*scale);}
	vector2d operator/(const real &div) const
		{real scale=1.0/div;return vector2d(x*scale,y*scale);}
	vector2d operator-(void) const {return vector2d(-x,-y);}
	void operator+=(const vector2d &b) {x+=b.x;y+=b.y;}
	void operator-=(const vector2d &b) {x-=b.x;y-=b.y;}
	void operator*=(const real scale) {x*=scale;y*=scale;}
	void operator/=(const real div) {real scale=1.0/div;x*=scale;y*=scale;}

//Vector-specific operations
	//Return the square of the magnitude of this vector
	real magSqr(void) const {return x*x+y*y;}
	//Return the magnitude (length) of this vector
	real mag(void) const {return sqrt(magSqr());}
	
	//Return the square of the distance to the vector b
	real distSqr(const vector2d &b) const 
		{return (x-b.x)*(x-b.x)+(y-b.y)*(y-b.y);}
	//Return the distance to the vector b
	real dist(const vector2d &b) const {return sqrt(distSqr(b));}
	
	//Return the dot product of this vector and b
	real dot(const vector2d &b) const {return x*b.x+y*b.y;}
	//Return the cosine of the angle between this vector and b
	real cosAng(const vector2d &b) const {return dot(b)/(mag()*b.mag());}
	
	//Return the "direction" (unit vector) of this vector
	vector2d dir(void) const {return (*this)/mag();}

	//Return the CCW perpendicular vector
	vector2d perp(void) const {return vector2d(-y,x);}

	//Return this vector scaled by that
	vector2d &scale(const vector2d &b) {x*=b.x;y*=b.y;return *this;}
	
	//Return the largest coordinate in this vector
	real max(void) {return (x>y)?x:y;}
	//Make each of this vector's coordinates at least as big
	// as the given vector's coordinates.
	void enlarge(const vector2d &by)
	{if (by.x>x) x=by.x; if (by.y>y) y=by.y;}
};

#endif //__OSL_VECTOR2D_H

/*
 * The former cktimer.h
 *
 */ 

#ifndef CMK_THRESHOLD_TIMER
#define CMK_THRESHOLD_TIMER

/** Time a sequence of operations, printing out the
names and times of any operations that exceed a threshold. 

Use it with only the constructor and destructor like:
      void foo(void) {
          CkThresholdTimer t("foo");
	  ...
      }
(this times the whole execution of the routine,
all the way to t's destructor is called on function return)

Or, you can start different sections like:
      void bar(void) {
          CkThresholdTimer t("first");
	  ...
	  t.start("second");
	  ...
	  t.start("third");
	  ...
      }
  
This class *only* prints out the time if it exceeds
a threshold-- by default, one millisecond.
*/
class CkThresholdTimer {
	double threshold; // Print any times that exceed this (s).
	double lastStart; // Last activity started at this time (s).
	const char *lastWhat; // Last activity has this name.
	
	void start_(const char *what) {
		lastStart=CmiWallTimer();
		lastWhat=what;
	}
	void done_(void) {
		double elapsed=CmiWallTimer()-lastStart;
		if (elapsed>threshold) {
			CmiPrintf("%s took %.2f s\n",lastWhat,elapsed);
		}
	}
public:
	CkThresholdTimer(const char *what,double thresh=0.001) 
		:threshold(thresh) { start_(what); }
	void start(const char *what) { done_(); start_(what); }
	~CkThresholdTimer() {done_();}
};

#endif
// end cktimer or ParFUM_timer



#endif
// end ParFUM_internals.h


