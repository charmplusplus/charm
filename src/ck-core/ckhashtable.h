/* Hash Table routines
Orion Sky Lawlor, olawlor@acm.org, 11/21/1999

This file defines the interface for the C++ Hashtable
class.  A Hashtable stores pointers to arbitrary "void *" 
objects so that they can be accessed in constant time.

This is a simple Hashtable implementation, with the interface
shamelessly stolen from java.util.Hashtable.  It dynamically
rehashes when the table gets too full.
*/

#ifndef __OSL_HASH_TABLE
#define __OSL_HASH_TABLE

//This data type is used to index into the hash table.
// For best results, all the bits of the hashCode should be
// meaningful (especially the high bits).
typedef unsigned int hashCode;

////////////////////////// HashKey (and children) ///////////////////////
//This is the base class of objects that can be used as keys into hash tables.

class HashKey {
protected:
	HashKey() {}//Keeps hashkeys from being built from outside
public:
	virtual ~HashKey();//Destructor

	//Return an arbitrary but repeatable number as a hash table index.
	//Computes a hash function of the key data.
	hashCode getHashCode(void) const;
	
	//Return 1 if this key equals the given key.
	//Compares the key length and data.
	int equals(const HashKey &that) const;
	
	// This method returns the length of and a pointer to the key data.
	//The returned pointer must be aligned to at least an integer boundary.
	virtual const unsigned char *getKey(/*out*/ int &len) const =0;
};

///////////////////////// HashTable //////////////////////


typedef HashKey *HashKeyPtr;
typedef void *HashObjectPtr;

class HashtableIterator;

class Hashtable {
protected:
	int tableSize;//Dimentions of arrays below (should be prime)
	int nElem;//Number of pointers actually stored in arrays below
	HashKeyPtr *keyTable;//Array of pointers to hash keys
	HashObjectPtr *objTable;//Array of pointers to hashed objects
	float loadFactor;//Maximum raction of table to fill 
	// (0->always enlarge; 1->only when absolutely needed)
	
	//Return the index into the table of the given key.
	//If the key is not in the table, but there is room for it,
	// returns the index where the key would go.  If no room, returns -1.
	static int findIndex(const HashKey &key,int tableSize,const HashKeyPtr *table);
	
	//Set the table array to the given size, re-hashing everything.
	void rehash(int newSize);
public:
	//Constructor-- create an empty hash table of at least the given size
	Hashtable(int initSize=17,float NloadFactor=0.75);
	//Destructor-- destroy keys and tables
	~Hashtable();
	
	//Add the given object to this table under the given key
	// The former object (if any) will be returned.
	// Table will be resized if needed.
	void *put(const HashKey &key,void *obj);
	
	//Look up the given object in this table.  Return NULL if not found.
	void *get(const HashKey &key) const;
	
	//Remove this object from the hashtable (re-hashing if needed)
	// Returns the old object, if there was one.
	void *remove(const HashKey &key);
	
	//Return an iterator for the objects in this hash table
	HashtableIterator *objects(void);
};

//A HashtableIterator lets you easily list all the objects
// in a hash table (without knowing all the keys).
class HashtableIterator {
protected:
	HashKeyPtr *keyTable;
	HashObjectPtr *objTable;
	int tableLen;//Length of table
	int curNo;//Table index of current object (to be returned next)
public:
	HashtableIterator(HashKeyPtr *NkeyTable,
		HashObjectPtr *NobjTable,
		int nLen);
	
	//Seek to start of hash table
	void seekStart(void);
	
	//Seek forward (or back) n hash slots
	void seek(int n);
	
	//Return 1 if next will be non-NULL
	int hasNext(void);
	//Return the next object, or NULL if none.
	// The corresponding object key will be returned in retKey.
	void *next(HashKey **retKey=NULL);
};

#endif /*__OSL_HASH_TABLE*/
