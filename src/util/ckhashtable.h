/* Hash Table routines
Orion Sky Lawlor, olawlor@acm.org, 11/21/1999

This file defines the interface for the C++ Hashtable
class.  A Hashtable stores key/object pairs
so that an arbitrary key can be accessed in 
constant time.

This is a simple Hashtable implementation, with the interface
shamelessly stolen from java.util.Hashtable.  It dynamically
rehashes when the table gets too full.  Both the key and
object are treated as arbitrary fixed-length runs of bytes.

Key hashing and comparison are handled by function pointers,
so the keys can be interpreted any way you like.  The default
(and C interface way) is to treat them as runs of plain bytes.
*/

#ifdef __cplusplus
extern "C" {
#endif
#ifndef __OSL_HASH_TABLE_C
#define __OSL_HASH_TABLE_C
/*C Version of Hashtable header file: */
typedef void *CkHashtable_c;

/*Create hashtable with a single integer as the key*/
CkHashtable_c CkCreateHashtable_int(int objBytes,int initSize);
/*Create hashtable with a C string pointer as the key */
CkHashtable_c CkCreateHashtable_string(int objBytes,int initSize);
/*Generic create*/
CkHashtable_c CkCreateHashtable(int keyBytes,int objBytes,int initSize);
void CkDeleteHashtable(CkHashtable_c h);

/*Return object storage for this (possibly new) key*/
void *CkHashtablePut(CkHashtable_c h,const void *atKey);

/*Return object storage for this (old) key-- */
/*  returns NULL if key not found.*/
void *CkHashtableGet(CkHashtable_c h,const void *fromKey);

/*Remove this key, rehashing as needed */
void CkHashtableRemove(CkHashtable_c h,const void *doomedKey);

#endif /*__OSL_HASH_TABLE_C*/
#ifdef __cplusplus
};

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
	return (h<<by)|(h>>(intBits-by));
}

//Functions to map keys to hash codes.
typedef CkHashCode (*CkHashFunction)(const void *keyData,size_t keyLen);
CkHashCode CkHashFunction_default(const void *keyData,size_t keyLen);
CkHashCode CkHashFunction_string(const void *keyData,size_t keyLen);
inline CkHashCode CkHashFunction_int(const void *keyData,size_t /*len*/)
	{return *(int *)keyData;}

//Functions return 1 if two keys are equal; 0 otherwise
typedef int (*CkHashCompare)(const void *key1,const void *key2,size_t keyLen);
int CkHashCompare_default(const void *key1,const void *key2,size_t keyLen);
int CkHashCompare_string(const void *key1,const void *key2,size_t keyLen);
inline int CkHashCompare_int(const void *k1,const void *k2,size_t /*len*/)
	{return *(int *)k1 == *(int *)k2;}

///////////////////////// Hashtable //////////////////////

class CkHashtableIterator;

class CkHashtable {
private:
	CkHashtable(const CkHashtable &); //Don't use these
	void operator=(const CkHashtable &);
protected:
	int len;//Vertical dimention of below array (best if prime)
	unsigned short kb,ob;//Bytes per key; bytes per object. eb=kb+ob
	unsigned short eb; //Bytes per hash entry-- horizontal dimention of below array
	typedef char *entry_t;
	entry_t *table;//Storage for keys and objects (len rows; eb columns)
	
	int nObj;//Number of objects actually stored (0..len)
	int resizeAt;//Resize when nObj>=resizeAt
	CkHashFunction hash;
	CkHashCompare compare;
	
	float loadFactor;//Maximum fraction of table to fill 
	// (0->always enlarge; 1->only when absolutely needed)
	
	//Increment i around the table
	int inc(int &i) const {i++; if (i>=len) i=0; return i;}
	
	//Return the start of the i'th entry in the hash table
	entry_t *entry(int i) const {return (entry_t *)(table+i*eb);}
	
	//Find the given key in the table.  If it's not there, return NULL
	entry_t *findKey(const void *key) const;
	//Find a spot for the given key in the table.  If there's not room, return NULL
	entry_t *findEntry(const void *key) const;
	
	//Build a new empty table of the given size
	void buildTable(int newLen);
	//Set the table to the given size, re-hashing everything.
	void rehash(int newLen);
public:
	//Constructor-- create an empty hash table of the given size
	CkHashtable(
		int keyBytes,int objBytes,//Bytes per key, object
		int initLen=5,float NloadFactor=0.75,
		CkHashFunction hash=CkHashFunction_default,
		CkHashCompare compare=CkHashCompare_default);
	//Destructor-- destroy table
	~CkHashtable();
	
	//Add the given object to this table under the given key
	// Returns pointer to object storage.
	// Table will be resized if needed.
	void *put(const void *key);
	
	//Look up the given object in this table.  Return NULL if not found.
	void *get(const void *key) {
		entry_t *r=findKey(key);
		if (r==NULL) return NULL;
		else return (void *)(r+kb);
	}
	
	//Remove this object from the hashtable (re-hashing if needed)
	void remove(const void *key);

	//Remove all objects and keys
	void empty(void);
	
	int numObjects(void) const { return nObj; } 
	
	//Return an iterator for the objects in this hash table
	CkHashtableIterator *iterator(void);
};

//A HashtableIterator lets you easily list all the objects
// in a hash table (without knowing all the keys).
class CkHashtableIterator {
protected:
	typedef char *entry_t;
	entry_t *table;
	int len; short kb,ob,eb;
	int curNo;//Table index of current object (to be returned next)
	//Return the start of the i'th entry in the hash table
	entry_t *entry(int i) const {return (entry_t *)(table+i*eb);}
public:
	CkHashtableIterator(void *Ntable,int Nlen,
		int Nkb,int Nob) :table((entry_t *)Ntable),len(Nlen),kb(Nkb),ob(Nob) 
		{curNo=0;eb=kb+ob;}
	
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
/*
This class provides a thin typesafe layer over the (unsafe)
CkHashtable above.  Via the magic of function inlining, this
comes at zero time and space cost over the unsafe version.
The unsafe version exists to avoid the code bloat associated
with profligate use of templates.
*/

template <class KEY, class OBJ> 
class CkHashtableTslow:public CkHashtable {
public:
	//Constructor-- create an empty hash table of at least the given size
	CkHashtableTslow(
		int initLen=5,float NloadFactor=0.5,
		CkHashFunction Nhash=CkHashFunction_default,
		CkHashCompare Ncompare=CkHashCompare_default)
	 :CkHashtable(sizeof(KEY),sizeof(OBJ),initLen,NloadFactor,Nhash,Ncompare)
	 {}
	
	OBJ &put(const KEY &key) {return *(OBJ *)CkHashtable::put((const void *)&key);}
	OBJ get(const KEY &key) {
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
// Be sure the hash and compare functions return the same
//  values as the non-inlined versions passed to the constructor.
template <class KEY, class OBJ> 
class CkHashtableT:public CkHashtableTslow<KEY,OBJ> {
public:
	//Constructor-- create an empty hash table of at least the given size
	CkHashtableT(
		int initLen=5,float NloadFactor=0.5,
		CkHashFunction Nhash=KEY::staticHash,
		CkHashCompare Ncompare=KEY::staticCompare)
		:CkHashtableTslow<KEY,OBJ>(initLen,NloadFactor,Nhash,Ncompare)
	 {}
	
	OBJ get(const KEY &key) {
		int i=key.hash()%len;
		while(true) {//Assumes key or empty slot will be found
			entry_t *cur=table+i*(sizeof(KEY)+sizeof(OBJ));
			//Is this the key?
			if (key.compare(*(const KEY *)cur))
				return *(OBJ *)(cur+sizeof(KEY));
			//An empty slot indicates the key is not here
			if (-17==*(int *)cur) return OBJ(0);
			inc(i);
		};
	}
	//Use this version when you're sure the entry exists
	OBJ &getRef(const KEY &key) {
		int i=key.hash()%len;
		while(true) {//Assumes key or empty slot will be found
			entry_t *cur=table+i*(sizeof(KEY)+sizeof(OBJ));
			//Is this the key?
			if (key.compare(*(const KEY *)cur))
				return *(OBJ *)(cur+sizeof(KEY));
			inc(i);
		};
	}
};
/*A useful adaptor class for using basic (memory only)
  types like int, short, char, etc. as hashtable keys.
 */
template <class T> class CkHashtableAdaptorT {
	T val;
public:
	CkHashtableAdaptorT<T>(const T &v):val(v) {}
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
};

#endif /*__OSL_HASH_TABLE_C++*/

#endif /*C++*/
