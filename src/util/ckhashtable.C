/* Hash Table routines
Orion Sky Lawlor, olawlor@acm.org, 11/21/1999

This file defines the interface for the C++ Hashtable
class.  A Hashtable stores pointers to arbitrary Hashable
objects so that they can be accessed in constant time.

This is a simple Hashtable implementation, with the interface
shamelessly stolen from java.util.Hashtable.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ckhashtable.h"

#include "converse.h"
#define DEBUGF(x) /*CmiPrintf x;*/

///////////////////// Default hash/compare functions ////////////////////
CkHashCode CkHashFunction_default(const void *keyData,size_t keyLen)
{
	const unsigned char *d=(const unsigned char *)keyData;
	CkHashCode ret=0;
	for (unsigned int i=0;i<keyLen;i++) {
		int shift1=((5*i)%16)+0;
		int shift2=((6*i)%16)+8;
		ret+=((0xa5^d[i])<<shift2)+(d[i]<<shift1);
	}
	DEBUGF(("    hashing %d-byte key to %08x\n",keyLen,ret))
	return ret;
}
CkHashCode CkHashFunction_string(const void *keyData,size_t keyLen)
{
	const char *d=*(const char **)keyData;
	CkHashCode ret=0;
	for (int i=0;d[i]!=0;i++) {
		int shift1=((5*i)%16)+0;
		int shift2=((6*i)%16)+8;
		ret+=((0xa5^d[i])<<shift2)+(d[i]<<shift1);
	}
	DEBUGF(("    hashed key '%s' to %08x\n",d,ret))
	return ret;
}

//Function to indicate when two keys are equal
int CkHashCompare_default(const void *key1,const void *key2,size_t keyLen)
{
	DEBUGF(("    comparing %d-byte keys--",keyLen))
	const char *a=(const char *)key1;
	const char *b=(const char *)key2;
	for (unsigned int i=0;i<keyLen;i++)
		if (a[i]!=b[i]) {DEBUGF(("different\n")) return 0;}
	DEBUGF(("equal\n"))
	return 1;
}
int CkHashCompare_string(const void *key1,const void *key2,size_t keyLen)
{
	const char *a=*(const char **)key1;
	const char *b=*(const char **)key2;
	DEBUGF(("    comparing '%s' and '%s'--",a,b))
	while (*a && *b)
		if (*a++!=*b++) {DEBUGF(("different\n")) return 0;}
	DEBUGF(("equal\n"))
	return 1;
}


/////////////////////// Hashtable implementation ///////////////
static unsigned int primeLargerThan(unsigned int x);
#define copyKey(dest,src) memcpy(dest,src,layout.keySize())
#define copyObj(dest,src) memcpy(dest,src,layout.objectSize())
#define copyEntry(dest,src) memcpy(dest,src,layout.entrySize())

////////////////////////// Hash Utility routines //////////////////////////

//Find the given key in the table.  If it's not there, return NULL
char *CkHashtable::findKey(const void *key) const
{
	DEBUGF(("  Finding key in table of %d--\n",len))
	int i=hash(key,layout.keySize())%len;
	int startSpot=i;
	do {
		char *cur=entry(i);
		if (layout.isEmpty(cur)) return NULL;
		char *curKey=layout.getKey(cur);
		if (compare(key,curKey,layout.keySize())) return curKey;
		DEBUGF(("   still looking for key (at %d)\n",i))
	} while (inc(i)!=startSpot);
	DEBUGF(("  No key found!\n"))
	return NULL;//We've searched the whole table-- no key.
}

//Find a spot for the given key in the table.  If there's not room, return NULL
char *CkHashtable::findEntry(const void *key) const
{
	DEBUGF(("  Finding spot in table of %d--\n",len))
	int i=hash(key,layout.keySize())%len;
	int startSpot=i;
	do {
		char *cur=entry(i);
		if (layout.isEmpty(cur)) return cur; //Empty spot
		char *curKey=layout.getKey(cur);
		if (compare(key,curKey,layout.keySize())) return cur; //Its old spot
		DEBUGF(("   still looking for spot (at %d)\n",i))
	} while (inc(i)!=startSpot);
	CmiAbort("  No spot found!\n");
	return NULL;//We've searched the whole table-- no room!
}

//Build a new empty table of the given size
void CkHashtable::buildTable(int newLen)
{
	len=newLen;
	resizeAt=(int)(len*loadFactor);
	DEBUGF(("Building table of %d (resize at %d)\n",len,resizeAt))
	table=new char[layout.entrySize()*len];
	for (int i=0;i<len;i++) layout.empty(entry(i));
}

//Set the table to the given size, re-hashing everything.
void CkHashtable::rehash(int newLen)
{
	DEBUGF(("Beginning rehash from %d to %d--\n",len,newLen))
	char *oldTable=table; //Save the old table
	int oldLen=len;
	buildTable(newLen); //Make a new table
	for (int i=0;i<oldLen;i++) {//Add all the old entries to the new table
		char *src=oldTable+i*layout.entrySize();
		if (!layout.isEmpty(src)) {
		  //There was an entry here-- copy it to the new table
		  char *dest=findEntry(layout.getKey(src));
		  copyEntry(dest,src);
		}
	}
	delete[] oldTable;
	DEBUGF(("Rehash complete\n"))
}

///////////////////////// Hashtable Routines //////////////////////

//Constructor-- create an empty hash table of at least the given size
CkHashtable::CkHashtable(const CkHashtableLayout &layout_,
	int initLen,float NloadFactor,
	CkHashFunction Nhash, //Maps keys to CkHashCodes
	CkHashCompare Ncompare)
  :layout(layout_)
{
	nObj=0;
	hash=Nhash;
	compare=Ncompare;
	loadFactor=NloadFactor;
	buildTable(initLen); //sets table, len, nObj, resizeAt
}

//Destructor-- destroy table
CkHashtable::~CkHashtable()
{
	DEBUGF(("Deleting table of %d\n",len))
	delete[] table;
	len=-1;
	nObj=-1;
}

//Remove all keys and objects
void CkHashtable::empty(void)
{
	for (int i=0;i<len;i++) {
		char *dest=entry(i);
		layout.empty(dest);
	}
	nObj = 0;
}

//Add the given object to this table under the given key
// Returns pointer to object storage.
// Table will be resized if needed.
void *CkHashtable::put(const void *key, int *existing)
{
	DEBUGF(("Putting key\n"))
#if 0
/*Check to make sure this table is consistent*/
	  int nActualObj=0;
	  for (int i=0;i<len;i++)
	    if (!layout.isEmpty(entry(i)))
	      nActualObj++;
	  if (nActualObj!=nObj) CmiAbort("Table corruption!\n");
#endif
	if (nObj>=resizeAt) rehash(primeLargerThan(len));
	char *ent=findEntry(key);
	if (layout.isEmpty(ent))
	{//Filling a new entry (*not* just replacing old one)
		nObj++;
		copyKey(layout.getKey(ent),key);
		layout.fill(ent);
		if (existing != NULL) *existing = 0;
	} else {
	  if (existing != NULL) *existing = 1;
	}
	return layout.getObject(ent);
}

/* Remove this object from the hashtable (re-hashing if needed)
   Returns the number of keys removed (always 0 or 1) */
int CkHashtable::remove(const void *key)
{
	DEBUGF(("Asked to remove key\n"))
	char *doomedKey=findKey(key);
	if (doomedKey==NULL) return 0; //It's already gone!
	nObj--;
	char *doomed=layout.entryFromKey(doomedKey);
	layout.empty(doomed);
	//Figure out where that entry came from in the table:
#define e2i(entry) (((entry)-table)/layout.entrySize())
	int i=e2i(doomed);
	DEBUGF(("Remove-rehashing later keys\n"))
	while (1) {
		inc(i);
		char *src=entry(i);
		if (layout.isEmpty(src))
		{//Stop once we find an empty key
			DEBUGF(("Remove-rehash complete\n"))
			return 1;
		}
		//This was a valid entry-- figure out where it goes now
		char *dest=findEntry(layout.getKey(src));
		if (src!=dest) {
			DEBUGF(("Remove-rehashing %d to %d\n",e2i(src),e2i(dest)))
			copyEntry(dest,src);
			layout.empty(src);
		} else {
			DEBUGF(("Remove-rehashing not needed for %d\n",e2i(src)))
		}
	}
}

/* Return an iterator for the objects in this hash table
 ** WARNING!!! ** This is a newly allocated memory that must be freed by the
 user with "delete" */
CkHashtableIterator *CkHashtable::iterator(void)
{
	DEBUGF(("Building iterator\n"))
	return new CkHashtableIterator(table,len,layout);
}

//////////////////////// HashtableIterator //////////////////
//A HashtableIterator lets you easily list all the objects
// in a hash table (without knowing all the keys).

//Seek to start of hash table
void CkHashtableIterator::seekStart(void) {curNo=0;}

//Seek forward (or back) n hash slots
void CkHashtableIterator::seek(int n)
{
  curNo+=n;
  if (curNo<0) curNo=0;
  if (curNo>len) curNo=len;
}

//Return 1 if next will be non-NULL
int CkHashtableIterator::hasNext(void)
{
  while (curNo<len) {
    if (!layout.isEmpty(entry(curNo)))
      return 1;//We have a next object
    else
      curNo++;//This spot is blank-- skip over it
  }
  return 0;//We went through the whole table-- no object
}

//Return the next object, or NULL if none
// The corresponding object key will be returned in retKey.
void *CkHashtableIterator::next(void **retKey)
{
  while (curNo<len) {
    char *cur=entry(curNo++);
    if (!layout.isEmpty(cur)) {
      //Here's the next object
      if (retKey) *retKey=layout.getKey(cur);
      return layout.getObject(cur);
    }
  }
  return NULL;//We went through the whole table-- no object
}

/************************ Prime List ************************
A smattering of prime numbers from 3 to 3 billion-billion.
I've chosen each one so it is approximately twice
the previous value.  Useful for hash table sizes, etc.

Orion Sky Lawlor, olawlor@acm.org, 11/18/1999
*/

const static unsigned int doublingPrimes[] = {
3,
7,
17,
37,
73,
157,
307,
617,
1217,
2417,
4817,
9677,
20117,
40177,
80177,
160117,
320107,
640007,
1280107,
2560171,
5120117,
10000079,
20000077,
40000217,
80000111,
160000177,
320000171,
640000171,
1280000017,
2560000217u,
4200000071u
/* extra primes larger than an unsigned 32-bit integer:
51200000077,
100000000171,
200000000171,
400000000171,
800000000117,
1600000000021,
3200000000051,
6400000000081,
12800000000003,
25600000000021,
51200000000077,
100000000000067,
200000000000027,
400000000000063,
800000000000017,
1600000000000007,
3200000000000059,
6400000000000007,
12800000000000009,
25600000000000003,
51200000000000023,
100000000000000003,
200000000000000003,
400000000000000013,
800000000000000119,
1600000000000000031,
3200000000000000059 //This is a 62-bit number
*/
};

//This routine returns an arbitrary prime larger than x
static unsigned int primeLargerThan(unsigned int x)
{
	int i=0;
	while (doublingPrimes[i]<=x)
		i++;
	return doublingPrimes[i];
}

/*************** C Interface routines ****************/
#define CDECL extern "C"

/*Create hashtable with a single integer as the key*/
CDECL CkHashtable_c CkCreateHashtable_int(int objBytes,int initSize)
{
  int objStart=2*sizeof(int);
  CkHashtableLayout layout(sizeof(int),sizeof(int),
			   objStart,objBytes,objStart+objBytes);
  return (CkHashtable_c)new CkHashtable(layout,initSize,0.5,
					CkHashFunction_int,CkHashCompare_int);
}
/*Create hashtable with a C string pointer as the key*/
CDECL CkHashtable_c CkCreateHashtable_string(int objBytes,int initSize)
{
  int objStart=2*sizeof(char *);
  CkHashtableLayout layout(sizeof(char *),sizeof(char *),
			   objStart,objBytes,objStart+objBytes);
  return (CkHashtable_c)new CkHashtable(layout,initSize,0.5,
					CkHashFunction_string,CkHashCompare_string);
}
/*Create hashtable with a C pointer as the key*/
CDECL CkHashtable_c CkCreateHashtable_pointer(int objBytes,int initSize)
{
  int objStart=2*sizeof(char *);
  CkHashtableLayout layout(sizeof(char *),sizeof(char *),
			   objStart,objBytes,objStart+objBytes);
  return (CkHashtable_c)new CkHashtable(layout,initSize,0.5,
					CkHashFunction_pointer,CkHashCompare_pointer);
}
CDECL void CkDeleteHashtable(CkHashtable_c h)
{
	delete (CkHashtable *)h;
}
/*Return object storage for this (possibly new) key*/
CDECL void *CkHashtablePut(CkHashtable_c h,const void *atKey)
{
	return ((CkHashtable *)h)->put(atKey);
}
/*Return object storage for this (old) key-- */
/*  returns NULL if key not found.*/
CDECL void *CkHashtableGet(CkHashtable_c h,const void *fromKey)
{
	return ((CkHashtable *)h)->get(fromKey);
}
/* Remove this key, rehashing as needed
   Returns the number of keys removed (always 0 or 1) */
CDECL int CkHashtableRemove(CkHashtable_c h,const void *doomedKey)
{
	return ((CkHashtable *)h)->remove(doomedKey);
}
/*Number of elements stored in the hashtable */
CDECL int CkHashtableSize(CkHashtable_c h)
{
    return ((CkHashtable *)h)->numObjects();
}

/*Return the iterator for the given hashtable. It is reset to the beginning.
 ** WARNING!!! ** This is a newly allocated memory that must be freed by the
 user with CkHashtableDestroyIterator */
CDECL CkHashtableIterator_c CkHashtableGetIterator(CkHashtable_c h) {
    CkHashtableIterator *it = ((CkHashtable *)h)->iterator();
    it->seekStart();
    return it;
}
/* Destroy the iterator allocated with CkHashtableGetIterator */
CDECL void CkHashtableDestroyIterator(CkHashtableIterator_c it) {
    delete ((CkHashtableIterator *)it);
}
/* Return the next element in the hash table given the iterator (NULL if not found) */
CDECL void *CkHashtableIteratorNext(CkHashtableIterator_c it, void **keyRet) {
    return ((CkHashtableIterator *)it)->next(keyRet);
}
/* Seek the iterator into the hashtable by 'n' slot (*not* objects) */
CDECL void CkHashtableIteratorSeek(CkHashtableIterator_c it, int n) {
    ((CkHashtableIterator *)it)->seek(n);
}
/* Seek the iterator into the hashtable by 'n' slot (*not* objects) */
CDECL void CkHashtableIteratorSeekStart(CkHashtableIterator_c it) {
    ((CkHashtableIterator *)it)->seekStart();
}
