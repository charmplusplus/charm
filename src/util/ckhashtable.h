/* Hash Table routines
   Orion Sky Lawlor, olawlor@acm.org, 11/21/1999

   This file defines the interface for the C++ Hashtable
   class.  A Hashtable stores key/object pairs
   so that an arbitrary key can be accessed in 
   constant time.

   This is a complicated non-chaining Hashtable implementation.
   It dynamically rehashes when the table gets too full.  
   Both the key and object are treated as arbitrary fixed-length 
   runs of bytes.

   Key hashing and comparison are handled by function pointers,
   so the keys can be interpreted any way you like.  The default
   (and C interface way) is to treat them as runs of plain bytes.
   */

#ifdef __cplusplus
#include "pup.h"
extern "C" {
#endif
#ifndef __cplusplus
#include "pup_c.h"
#endif

#ifndef __OSL_HASH_TABLE_C
#define __OSL_HASH_TABLE_C

#include <stddef.h>

    /*C Version of Hashtable header file: */
    typedef void *CkHashtable_c;

    /*Create hashtable with a single integer as the key*/
    CkHashtable_c CkCreateHashtable_int(int objBytes,int initSize);
    /*Create hashtable with a C string pointer as the key */
    CkHashtable_c CkCreateHashtable_string(int objBytes,int initSize);
    /*Create hashtable with a C pointer as the key */
    CkHashtable_c CkCreateHashtable_pointer(int objBytes,int initSize);

    void CkDeleteHashtable(CkHashtable_c h);

    /*Return object storage for this (possibly new) key*/
    void *CkHashtablePut(CkHashtable_c h,const void *atKey);

    /*Return object storage for this (old) key-- */
    /*  returns NULL if key not found.*/
    void *CkHashtableGet(CkHashtable_c h,const void *fromKey);

    /*Return the key associated with this object (previously returned with one of the above functions) */
    void *CkHashtableKeyFromObject(CkHashtable_c h,const void *object);

    /* Remove this key, rehashing as needed.
       Returns the number of keys removed (always 0 or 1) */
    int CkHashtableRemove(CkHashtable_c h,const void *doomedKey);

    /*Number of elements stored in the hashtable */
    int CkHashtableSize(CkHashtable_c h);

    /*C Version of Hashtable iterator */
    typedef void *CkHashtableIterator_c;

    /*Return the iterator for the given hashtable. It is reset to the beginning
     ** WARNING!!! ** This is a newly allocated memory that must be freed by the
     user with CkHashtableDestroyIterator */
    CkHashtableIterator_c CkHashtableGetIterator(CkHashtable_c h);

    /* Destroy the iterator allocated with CkHashtableGetIterator */
    void CkHashtableDestroyIterator(CkHashtableIterator_c it);

    /* Return the next element in the hash table given the iterator (NULL if not found) */
    void *CkHashtableIteratorNext(CkHashtableIterator_c it, void **retKey);

    /* Seek the iterator into the hashtable by 'n' slot (*not* objects) */
    void CkHashtableIteratorSeek(CkHashtableIterator_c it, int n);

    /* Seek the iterator into the hashtable by 'n' slot (*not* objects) */
    void CkHashtableIteratorSeekStart(CkHashtableIterator_c it);

#endif /*__OSL_HASH_TABLE_C*/
#ifdef __cplusplus
}

#ifndef __OSL_HASH_TABLE_CPP
#define __OSL_HASH_TABLE_CPP

#include <stdio.h>

//This data type is used to index into the hash table.
// For best results, all the bits of the hashCode should be
// meaningful (especially the high bits).
typedef unsigned int CkHashCode;

//A circular-left-shift, useful for creating hash codes.
inline CkHashCode circleShift(CkHashCode h,unsigned int by) 
{
    const unsigned int intBits=8*sizeof(CkHashCode);
    by%=intBits;
    if (by == 0)
      return h;
    return (h<<by)|(h>>(intBits-by));
}

//Functions to map keys to hash codes.
typedef CkHashCode (*CkHashFunction)(const void *keyData,size_t keyLen);
CkHashCode CkHashFunction_default(const void *keyData,size_t keyLen);
CkHashCode CkHashFunction_string(const void *keyData,size_t keyLen);
inline CkHashCode CkHashFunction_int(const void *keyData,size_t /*len*/)
{return *(int *)keyData;}
inline CkHashCode CkHashFunction_pointer(const void *keyData,size_t /*len*/)
{if (sizeof(char*)==sizeof(int)) return *(int *)keyData;
    else if (sizeof(char*)==2*sizeof(int)) return ((int*)keyData)[0] ^ ((int*)keyData)[1];
    else CmiAbort("Invalid key data for hash code");
        return 0;
	}

//Functions return 1 if two keys are equal; 0 otherwise
typedef int (*CkHashCompare)(const void *key1,const void *key2,size_t keyLen);
int CkHashCompare_default(const void *key1,const void *key2,size_t keyLen);
int CkHashCompare_string(const void *key1,const void *key2,size_t keyLen);
inline int CkHashCompare_int(const void *k1,const void *k2,size_t /*len*/)
	{return *(int *)k1 == *(int *)k2;}
inline int CkHashCompare_pointer(const void *k1,const void *k2,size_t /*len*/)
	{return *(char **)k1 == *(char **)k2;}


///////////////////////// Hashtable //////////////////////

class CkHashtableIterator;

/**
 Describes the in-memory layout of a hashtable entry.

Nobody should ever use this class directly; use CkHashtableT
or CkHashtableTslow instead.

"key" is a user-defined type, used as the unique object identifier.
The key is assumed to begin at the start of the entry.  
"empty" is a character, set to 1 if this entry in the table 
is unused, zero otherwise; the "empty" field must not overlap 
either of the other fields. "object" is the thing the table stores;
it is of a user-defined type and may overlap "key".

     | key | empty | gap? | object | gap? |
==   | <------- hashtable entry --------> |

*/
class CkHashtableLayout {
  int size; ///< Size of entire table entry, at least ks+os.
  int ko,ks; ///< Key byte offset (always zero) and size
  int po,ps; ///< "empty bit" offset and size (always 1)
  int oo,os; ///< Object byte offset and size
 public:
  CkHashtableLayout(int keySize,int emptyOffset,
		    int objectOffset,int objectSize,int entryLength):
		size(entryLength),
		ko(0), ks(keySize),
		po(emptyOffset), ps(1),
		oo(objectOffset), os(objectSize)
  {}

  inline int entrySize(void) const {return size;}
  inline int keySize(void) const {return ks;}
  inline int objectSize(void) const {return os;}

//Utility functions:
  /// Given an entry pointer, return a pointer to the key
  inline char *getKey(char *entry) const {return entry+ko;}
  /// Given an entry pointer, return a pointer to the object
  inline char *getObject(char *entry) const {return entry+oo;}
  
  /// Is this entry empty?
  inline char isEmpty(char *entry) const {return *(entry+po);}
  /// Mark this entry as empty
  inline void empty(char *entry) const {*(entry+po)=1;}  
  /// Mark this entry as full
  inline void fill(char *entry) const {*(entry+po)=0;}

  /// Move to the next entry
  inline char *nextEntry(char *entry) const {return entry+size;}

  /// Get entry pointer from key pointer
  inline char *entryFromKey(char *key) const {return key-ko;}
  
  /// Get entry pointer from object pointer
  inline char *entryFromObject(char *obj) const {return obj-oo;}
};

/**
 A resize-on-demand extensible hashtable.  Users should probably
use CkHashtableT or CkHashtableTslow instead of calling this class
directly.
*/
class CkHashtable {
private:
	CkHashtable(const CkHashtable &); //Don't use these
	void operator=(const CkHashtable &);
protected:
	int len;//Vertical dimension of below array (best if prime)
	CkHashtableLayout layout; //Byte-wise storage layout for an entry
	char *table;//len hashtable entries
	
	int nObj;//Number of objects actually stored (0..len)
	int resizeAt;//Resize when nObj>=resizeAt
	CkHashFunction hash; //Function pointer to compute key hash
	CkHashCompare compare; //Function pointer to compare keys
	
	float loadFactor;//Maximum fraction of table to fill 
	// (0->always enlarge; 1->only when absolutely needed)
	
	//Increment i around the table
	int inc(int &i) const {i++; if (i>=len) i=0; return i;}
	
	//Return the start of the i'th entry in the hash table
	char *entry(int i) const {return (char *)(table+i*layout.entrySize());}

	//Find the given key in the table.  If it's not there, return NULL
	char *findKey(const void *key) const;
	//Find a spot for the given key in the table.  If there's not room, return NULL
	char *findEntry(const void *key) const;
	
	//Build a new empty table of the given size
	void buildTable(int newLen);
	//Set the table to the given size, re-hashing everything.
	void rehash(int newLen);
public:
	//Constructor-- create an empty hash table of the given size
	CkHashtable(const CkHashtableLayout &layout_,
		int initLen=5,float NloadFactor=0.5,
		CkHashFunction hash=CkHashFunction_default,
		CkHashCompare compare=CkHashCompare_default);
	//Destructor-- destroy table
	~CkHashtable();
	
	//Add the given object to this table under the given key
	// Returns pointer to object storage.
	// Table will be resized if needed.
	void *put(const void *key, int *existing=NULL);
	
	//Look up the given object in this table.  Return NULL if not found.
	void *get(const void *key) const {
		char *ent=findKey(key);
		if (ent==NULL) return NULL;
		else return layout.getObject(ent);
	}
	
	/* Remove this object from the hashtable (re-hashing if needed)
       Returns the number of keys removed (always 0 or 1) */
	int remove(const void *key);

	//Remove all objects and keys
	void empty(void);
	
	int numObjects(void) const { return nObj; } 
	
	/* Return an iterator for the objects in this hash table.
	   ** WARNING!!! ** This is a newly allocated memory that must be freed by the
	   user with "delete" */
	CkHashtableIterator *iterator(void);
};

//A HashtableIterator lets you easily list all the objects
// in a hash table (without knowing all the keys).
class CkHashtableIterator {
protected:
	int len;
	CkHashtableLayout layout; //Byte-wise storage layout for an entry
	char *table;
	int curNo;//Table index of current object (to be returned next)
	//Return the start of the i'th entry in the hash table
	char *entry(int i) const {return table+i*layout.entrySize();}
public:
	CkHashtableIterator(char *table_,int len_,const CkHashtableLayout &lo)
	  :len(len_),layout(lo),table(table_),curNo(0)
		{}
        CkHashtableIterator(const CkHashtableIterator &it);  //don't copy
        void operator=(const CkHashtableIterator &it);	    // don't copy
	
	//Seek to start of hash table
	void seekStart(void);
	
	//Seek forward (or back) n hash slots (*not* n objects!)
	void seek(int n);
	
	//Return 1 if next will be non-NULL
	int hasNext(void);
	//Return the next object, or NULL if none.
	// The corresponding object key will be returned in retKey.
	void *next(void **retKey=NULL);
};


/////////////////// Templated Hashtable /////////////////
/**
This class provides a thin typesafe layer over the (unsafe)
CkHashtable above.  Via the magic of function inlining, this
comes at zero time and space cost over the unsafe version.

The unsafe version exists to avoid the code bloat associated
with profligate use of templates; because the typeless layer
handles the difficult parts, this templated class is very cheap.
*/
template <class KEY, class OBJ> 
class CkHashtableTslow:public CkHashtable {
  //Return the layout for this hashtable:
  static inline CkHashtableLayout getLayout(void) {
    //This struct defines the in-memory layout that we will use.
    //  By including it in a struct rather than figuring it out ourselves, 
    //  we let the compiler figure out what the (bizarre) alignment requirements are.
    struct entry_t {
      KEY k;
      char empty;
      OBJ o;
    };
    // HACK: All I want is the offset from entry_t to empty and o;
    //  but the compiler's "offsetof" keyword complains "non-POD type!".
    entry_t *e=(entry_t *)0;
    int emptyOffset=((char *)&e->empty)-(char *)e;
    int oOffset=((char *)&e->o)-(char *)e;
    return CkHashtableLayout(sizeof(KEY),emptyOffset,
			     oOffset,sizeof(OBJ),sizeof(entry_t));
  }
public:
	//Constructor-- create an empty hash table of at least the given size
	CkHashtableTslow(
		int initLen=5,float NloadFactor=0.5,
		CkHashFunction Nhash=CkHashFunction_default,
		CkHashCompare Ncompare=CkHashCompare_default)
	 :CkHashtable(getLayout(),initLen,NloadFactor,Nhash,Ncompare)
	 {}
	
	OBJ &put(const KEY &key, int *existing=NULL) {
		void *obj = CkHashtable::put((const void *)&key, existing);
		return *(OBJ *)obj;
	}
	OBJ get(const KEY &key) const {
		void *r=CkHashtable::get((const void *)&key);
		if (r==NULL) return OBJ(0);
		else return *(OBJ *)r;
	}
	//Use this version when you're sure the entry exists
	OBJ &getRef(const KEY &key) {
		return *(OBJ *)CkHashtable::get((const void *)&key);
	}
	void remove(const KEY &key) {CkHashtable::remove((const void *)&key);}
};

//Declare the KEY.hash & KEY.compare functions inline for a 
// completely inlined (very fast) hashtable lookup.	
// You MUST be SURE the hash and compare functions return the same
//  values as the staticHash and staticCompare functions.
template <class KEY, class OBJ> 
class CkHashtableT:public CkHashtableTslow<KEY,OBJ> {
public:
	//Constructor-- create an empty hash table of at least the given size
	CkHashtableT(int initLen=5,float NloadFactor=0.5)
	  :CkHashtableTslow<KEY,OBJ>(initLen,NloadFactor,
				     KEY::staticHash,KEY::staticCompare)
	{}
	CkHashtableT(
		int initLen,float NloadFactor,
		CkHashFunction Nhash,CkHashCompare Ncompare)
		:CkHashtableTslow<KEY,OBJ>(initLen,NloadFactor,Nhash,Ncompare)
	 {}
	
	//Return the object, or "0" if it doesn't exist
	OBJ get(const KEY &key) const {
		int i=key.hash()%this->len;
		while(1) {//Assumes key or empty slot will be found
			char *cur=this->entry(i);
			//An empty slot indicates the key is not here
			if (this->layout.isEmpty(cur)){ 
				return OBJ(0);
			}
			//Is this the key?
			if (key.compare(*(KEY *)this->layout.getKey(cur)))
				return *(OBJ *)this->layout.getObject(cur);
			this->inc(i);
		};
	}

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	OBJ *getPointer(const KEY &key) {
        int i=key.hash()%this->len;
        while(1) {//Assumes key or empty slot will be found
            char *cur=this->entry(i);
			//An empty slot indicates the key is not here
            if (this->layout.isEmpty(cur)){
                return NULL;
            }
			//Is this the key?
            if (key.compare(*(KEY *)this->layout.getKey(cur)))
                return (OBJ *)this->layout.getObject(cur);
            this->inc(i);
        };
        return NULL;
    }
#endif

	//Use this version when you're sure the entry exists--
	// avoids the test for an empty entry
	OBJ &getRef(const KEY &key) {
		int i=key.hash()%this->len;
		while(1) {//Assumes key or empty slot will be found
			char *cur=this->entry(i);
			//Is this the key?
			if (key.compare(*(KEY *)this->layout.getKey(cur)))
				return *(OBJ *)this->layout.getObject(cur);
			this->inc(i);
		};
	}
	void pup(PUP::er &p){
		if(!p.isUnpacking()){
			/*packing phase: loop through the hashtable values*/
			CkHashtableIterator *it=CkHashtable::iterator();
			int hasNext=1;
			OBJ *o; KEY *k;
			while (NULL!=(o=(OBJ *)it->next((void **)&k))) {
				p|hasNext;
				p|*k;
				p|*o;
			}
			hasNext=0; p|hasNext;
			delete it;
		}else{
		/*Unpacking phase: add each hashtable item*/
			int hasNext=1;
			p|hasNext;
			while (hasNext) {
				OBJ o; KEY k;
				p|k;
				p|o;
				this->put(k)=o;
				p|hasNext;
			}
		}
	}
};

/**
A useful adaptor class for using basic (memory only)
  types like int, short, char, etc. as hashtable keys.

This class adds the hash, compare, staticHash, and staticCompare
routines needed to be a CkHashtableT KEY; you can thus use ints
as a fast key like this:
	CkHashtableT<CkHashtableAdaptorT<int>, foo> bar;

*/
template <class T> class CkHashtableAdaptorT {
	T val;
public:
	CkHashtableAdaptorT<T>(const T &v):val(v) {}
	/**added to allow pup to do Key k while unPacking*/
	CkHashtableAdaptorT<T>(){}
	operator T & () {return val;}
	operator const T & () const {return val;}
	inline CkHashCode hash(void) const 
		{return (CkHashCode)val;}
	static CkHashCode staticHash(const void *k,size_t) 
		{return ((CkHashtableAdaptorT<T> *)k)->hash();}
	inline int compare(const CkHashtableAdaptorT<T> &t) const
		{return val==t.val;}
	static int staticCompare(const void *a,const void *b,size_t) 
	{
		return ((CkHashtableAdaptorT<T> *)a)->
		     compare(*(CkHashtableAdaptorT<T> *)b);
	}
	void pup(PUP::er &p){
		p | val;
	}
};

#endif /*__OSL_HASH_TABLE_C++*/

#endif /*C++*/
