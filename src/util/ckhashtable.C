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
#include "conv-autoconfig.h"
#include "ckhashtable.h"

#define DEBUGF(x) /*printf x;*/
#define ABORT(str) /*CkAbort(str)*/ do {printf(str);abort();} while (0)


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
#define copyKey(dest,src,len) memcpy(dest,src,len) //Move a key
#define copyObj(dest,src,len) memcpy(dest,src,len) //Move an object
#define keyEmpty(key) (*(int *)(key)==-17) //Is this key empty?
#define emptyKey(key) (*(int *)(key)=-17)  //Make this key empty
#define e2k(e) ((void *)(e))  //Entry -> key
#define e2o(e) ((void *)((e)+kb)) //Entry -> object
#define e2i(e) (((e)-table)/eb) //Entry -> index

////////////////////////// Hash Utility routines //////////////////////////

//Find the given key in the table.  If it's not there, return NULL
CkHashtable::entry_t *CkHashtable::findKey(const void *key) const
{
	DEBUGF(("  Finding key in table of %d--\n",len))
	int i=hash(key,kb)%len;
	int startSpot=i;
	do {
		entry_t *cur=entry(i);
		if (compare(key,e2k(cur),kb)) return cur;
		if (keyEmpty(e2k(cur))) return NULL;
		DEBUGF(("   still looking for key...\n"))
	} while (inc(i)!=startSpot);
	DEBUGF(("  No key found!\n"))
	return NULL;//We've searched the whole table-- no key.
}

//Find a spot for the given key in the table.  If there's not room, return NULL
CkHashtable::entry_t *CkHashtable::findEntry(const void *key) const
{
	DEBUGF(("  Finding spot in table of %d--\n",len))
	int i=hash(key,kb)%len;
	int startSpot=i;
	do {
		entry_t *cur=entry(i);
		if (keyEmpty(e2k(cur))) return cur;
		if (compare(key,e2k(cur),kb)) return cur;
		DEBUGF(("   still looking for spot...\n"))
	} while (inc(i)!=startSpot);
	DEBUGF(("  No spot found!\n"))
	return NULL;//We've searched the whole table-- no room!
}

//Build a new empty table of the given size
void CkHashtable::buildTable(int newLen)
{
	len=newLen;
	nObj=0;
	resizeAt=(int)(len*loadFactor);
	DEBUGF(("Building table of %d (resize at %d)\n",len,resizeAt))
	table=new entry_t[eb*len];
	for (int i=0;i<len;i++) {
		entry_t *dest=entry(i);
		emptyKey(e2k(dest));
	}
}

//Set the table to the given size, re-hashing everything.
void CkHashtable::rehash(int newLen)
{
	DEBUGF(("Beginning rehash from %d to %d--\n",len,newLen))
	entry_t *oldTable=table; //Save the old table
	int oldLen=len;
	buildTable(newLen); //Make a new table
	for (int i=0;i<oldLen;i++) {//Add all the old entries to the new table
		entry_t *e=oldTable+i*eb;
		if (!keyEmpty(e2k(e)))
			copyObj(put(e2k(e)),e2o(e),ob);
	}
	delete[] oldTable;
	DEBUGF(("Rehash complete\n"))
}

///////////////////////// Hashtable Routines //////////////////////

//Constructor-- create an empty hash table of at least the given size
CkHashtable::CkHashtable(
	int keyBytes,int objBytes,//Bytes per key, object
	int initLen,float NloadFactor,
	CkHashFunction Nhash, //Maps keys to CkHashCodes
	CkHashCompare Ncompare)
{
	kb=keyBytes;ob=objBytes;eb=kb+ob;
	
	/*Since we use int(-1) as the empty key sentinal, we require
	  enough space for the int and the records to be aligned. */
	if (eb<sizeof(int))
		ABORT("CkHashtable needs keyBytes+objBytes>=sizeof(int)!");
	if (eb/sizeof(int)*sizeof(int)!=eb)
		ABORT("CkHashtable needs sizeof(key+obj) to be a multiple of sizeof(int)!");
	
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

//Add the given object to this table under the given key
// Returns pointer to object storage.
// Table will be resized if needed.
void *CkHashtable::put(const void *key)
{
	DEBUGF(("Putting key\n"))
	if (nObj>=resizeAt) rehash(primeLargerThan(len));
	entry_t *ent=findEntry(key);
	if (keyEmpty(e2k(ent)))
	{//Filling new entry (*not* just replacing old one)
		nObj++;
		copyKey(e2k(ent),key,kb);
	}
	return e2o(ent);
}

//Remove this object from the hashtable (re-hashing if needed)
void CkHashtable::remove(const void *key)
{
	DEBUGF(("Asked to remove key\n"))
	entry_t *doomed=findKey(key);
	if (doomed==NULL) return;
	nObj--;
	emptyKey(e2k(doomed));
	int i=e2i(doomed);
	DEBUGF(("Remove-rehashing later keys\n"))
	while (1) {
		inc(i);
		entry_t *src=entry(i);
		if (keyEmpty(e2k(src))) 
		{//Stop once we find an empty key
			DEBUGF(("Remove-rehash complete\n"))
			return;
		}
		entry_t *dest=findEntry(e2k(src));
		if (src!=dest) {
			//Move src to dest; clear src
			DEBUGF(("Remove-rehashing %d to %d\n",e2i(src),e2i(dest)))
			copyKey(e2k(dest),e2k(src),kb);
			copyObj(e2o(dest),e2o(src),ob);
			emptyKey(e2k(src));
		} else {
			DEBUGF(("Remove-rehashing not needed for %d\n",e2i(src)))
		}
	}
}

//Return an iterator for the objects in this hash table
CkHashtableIterator *CkHashtable::iterator(void)
{
	DEBUGF(("Building iterator\n"))
	return new CkHashtableIterator(table,len,kb,ob);
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
    if (!keyEmpty(e2k(entry(curNo))))
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
    entry_t *cur=entry(curNo++);
    if (!keyEmpty(e2k(cur))) {
      //Here's the next object
      if (retKey) *retKey=e2k(cur);
      return e2o(cur);
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
/*Create generic hashtable (any sort of key)*/
CDECL CkHashtable_c CkCreateHashtable(int keyBytes,int objBytes,int initSize)
{
	return (CkHashtable_c)new CkHashtable(keyBytes,objBytes,initSize);
}
/*Create hashtable with a single integer as the key*/
CDECL CkHashtable_c CkCreateHashtable_int(int objBytes,int initSize)
{
	return (CkHashtable_c)new CkHashtable(sizeof(int),objBytes,initSize,0.75,
		CkHashFunction_int,CkHashCompare_int);
}
/*Create hashtable with a C string pointer as the key*/
CDECL CkHashtable_c CkCreateHashtable_string(int objBytes,int initSize)
{
	return (CkHashtable_c)new CkHashtable(sizeof(char *),objBytes,initSize,0.75,
		CkHashFunction_string,CkHashCompare_string);
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
/*Remove this key, rehashing as needed */
CDECL void CkHashtableRemove(CkHashtable_c h,const void *doomedKey)
{
	((CkHashtable *)h)->remove(doomedKey);
}
