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
#ifndef  WIN32
#include <values.h> //For INTBITS
#endif
#include "pup.h"
#include "ckhashtable.h"

static unsigned int primeLargerThan(unsigned int x);

////////////////////////// HashKey //////////////////////////
//This is the base class of objects that can be used as keys into hash tables.
HashKey::~HashKey() {} //Destructor does nothing by default

//Return an arbitrary but repeatable number as a hash table index.
//The default hashes the keyLen and keyData using a simple 
// XOR'd bitwise-rotation scheme.
hashCode HashKey::getHashCode(void) const
{
#ifndef INTBITS
# define INTBITS 32
#endif
//Utility define:
//return the value x cyclic-left-shifted (cls) by "by" bits modulo 32.
#define cls(x,by) ((((unsigned int)(x))<<(         (by) &(INTBITS-1)))| \
                   (((unsigned int)(x))>>((INTBITS-(by))&(INTBITS-1))))
        
	register hashCode ret=0;
	register int i,len;
	register const unsigned char *d=this->getKey(len);
	register const int *di=(const int *)d;
	
//Hash as much as possible using integers 
//(This is why the key data must be aligned on an integer boundary)
	int numInts=int(((unsigned int)len)/sizeof(int));
	for (i=0;i<numInts;i++)
		ret ^=cls(di[i],22+11*i);
	
//Now cleanup the last remaining bytes
	for (i=numInts*sizeof(int);i<len;i++)
		ret ^=cls(d[i],14+11*i);
	return ret;
}

//Return 1 if this HashKey equals that hashKey.
//The default compares the key length and data from above.
int HashKey::equals(const HashKey &that) const
{
	int thisLen,thatLen;
	const unsigned char *thisData=this->getKey(thisLen);
	const unsigned char *thatData=that.getKey(thatLen);
	const int *thisInt=(const int *)thisData;
	const int *thatInt=(const int *)thatData;
	
	if (thisLen!=thatLen) return 0;//These keys have different lengths!
	int i;
//Compare as much as possible using integers 
//(This is why the key data must be aligned on an integer boundary)
	int numInts=int(((unsigned int)thisLen)/sizeof(int));
	for (i=0;i<numInts;i++)
		if (thisInt[i]!=thatInt[i])
			return 0;//These keys have different data!
	
//Now compare the last remaining bytes
	for (i=numInts*sizeof(int);i<thisLen;i++)
		if (thisData[i]!=thatData[i])
			return 0;//These keys have different data!
	//If we got here, the two keys must have exactly the same data
	return 1;
}


// This hashkey is used to store a (variable) n bytes of data
class hashKeyHeap:public HashKey {
protected:
	int nBytes;unsigned char *data;
public:
	hashKeyHeap(int NnBytes,const void *Ndata) {
		nBytes=NnBytes;
		data=new unsigned char[nBytes];
		memcpy(data,Ndata,nBytes);
	}
	~hashKeyHeap() {delete data;}
	virtual const unsigned char *getKey(int &len) const {
		len=nBytes;
		return data;
	}
	virtual void pup(PUP::er &p) {
		p(nBytes);
		if (p.isUnpacking()) data = new unsigned char[nBytes];
		p((void *) data, nBytes);
	}
};

// This hashkey stores a (fixed) n bytes of data
template <int n>
class hashKeyFixed:public HashKey {
protected:
	//We declare data as an int array so it's int-aligned.
	int data[(n+sizeof(int)-1)/sizeof(int)];//<- n/sizeof(int), rounded up
public:
	hashKeyFixed(const void *Ndata) {
		memcpy((void *)&data,(const void *)Ndata,n);
	}
	virtual const unsigned char *getKey(int &len) const {
		len=n;
		return (const unsigned char *)&data;
	}
	virtual void pup(PUP::er &p) {
		int len = n;
		p(len);
		p((void *) data, n);
	}
};

// This hashkey stores no bytes of data
class hashKeyNone:public HashKey {
public:
	virtual const unsigned char *getKey(int &len) const
		{ len=0;return NULL; }
	virtual void pup(PUP::er &p) { int len = 0; p(len); }
};

//static method - Return a heap-allocated key containing the given data
HashKey *HashKey::newKey(int n,const void *data)
{
	switch(n)
	{
	case 0: return new hashKeyNone();
	case 1: return new hashKeyFixed<1>(data);
	case 2: return new hashKeyFixed<2>(data);
	case 3: return new hashKeyFixed<3>(data);
	case 4: return new hashKeyFixed<4>(data);
	case 5: return new hashKeyFixed<5>(data);
	case 6: return new hashKeyFixed<6>(data);
	case 7: return new hashKeyFixed<7>(data);
	case 8: return new hashKeyFixed<8>(data);
	case 9: return new hashKeyFixed<9>(data);
	case 10: return new hashKeyFixed<10>(data);
	case 11: return new hashKeyFixed<11>(data);
	case 12: return new hashKeyFixed<12>(data);
	case 13: return new hashKeyFixed<13>(data);
	case 14: return new hashKeyFixed<14>(data);
	case 15: return new hashKeyFixed<15>(data);
	case 16: return new hashKeyFixed<16>(data);
	case 17: return new hashKeyFixed<17>(data);
	case 18: return new hashKeyFixed<18>(data);
	case 19: return new hashKeyFixed<19>(data);
	case 20: return new hashKeyFixed<20>(data);
	default: return new hashKeyHeap(n,data);
	}
}

//Return a heap-allocated key containing my data
HashKey *HashKey::newKey(void) const
{
  int n;
  const unsigned char *d=getKey(n);
  return newKey(n,(const void *)d); //call static newKey to allocate key
}

///////////////////////// Hashtable //////////////////////

#define DEBUGF(x) /*printf x;*/

//static method:
//Return the index into the table of the given key.
//If the key is not in the table, but there is room for it,
// returns the index where the key would go.  If no room, returns -1.
int Hashtable::findIndex(const HashKey &key,int tableSize,const HashKeyPtr *table)
{
  DEBUGF(("Finding index in table of %d--\n",tableSize))
    hashCode code=key.getHashCode();
  int startIndex=code%tableSize;
  int nRemaining=tableSize;//Number of unsearched locations
  int index=startIndex;
  
  //Look for the key starting at the hash index
  while (nRemaining>0)
    {
      if (table[index]!=NULL)
	{
	  if (table[index]->equals(key)) 
	    return index; //We found the key!
	} else 
	  return index;//Here's an empty spot for the key
      index++;if (index>=tableSize) index=0;//Wrap around to start
      nRemaining--;
    }
  
  
  DEBUGF(("No index in table of size %d.\n",tableSize))
    //At this point, we've searched the entire array and 
    // haven't found the key *or* an empty spot for it.
    return -1;
}

//Set the table array to the given size, re-hashing everything.
void Hashtable::rehash(int newSize)
{
  DEBUGF(("Setting size to %d\n",newSize))
    int i;
  HashKeyPtr *newKeyTable=new HashKeyPtr[newSize];
  HashObjectPtr *newObjTable=new HashObjectPtr[newSize];
  
  //Zero out the new table
  for (i=0;i<newSize;i++)
    {newKeyTable[i]=NULL;newObjTable[i]=NULL;}
  
  //Copy the old keys into the new table
  for (i=0;i<tableSize;i++)
    if (keyTable[i]!=NULL)
      {//This key is set-- find a place for it in the new table & copy it
	int newIndex=findIndex(*keyTable[i],newSize,newKeyTable);
	newKeyTable[newIndex]=keyTable[i];
	newObjTable[newIndex]=objTable[i];
	DEBUGF(("Expansion re-hash: %d -> %d\n",i,newIndex)) 
	  }
  
  //Delete the old tables
  delete keyTable;
  delete objTable;
  
  //Copy over the new tables
  tableSize=newSize;
  keyTable=newKeyTable;
  objTable=newObjTable;
}

//Constructor-- create an empty hash table of at least the given size
Hashtable::Hashtable(int initSize,float NloadFactor)
{
  loadFactor=NloadFactor;
  tableSize=primeLargerThan(initSize-1);
  
  //Allocate a new table
  keyTable=new HashKeyPtr[tableSize];
  objTable=new HashObjectPtr[tableSize];
  
  //Zero out the new table
  for (int i=0;i<tableSize;i++)
    {keyTable[i]=NULL;objTable[i]=NULL;}
  nElem=0;
}

//Destructor-- destroy keys and tables
Hashtable::~Hashtable()
{
  for (int i=0;i<tableSize;i++)
    if (keyTable[i]!=NULL)
      delete keyTable[i];
  delete keyTable;
  delete objTable;
  nElem=-1;tableSize=-1;
}

//Add the given object to this table under the given key
// The former object (if any) will be returned.
// Table will be resized if needed.
void *Hashtable::put(const HashKey &key,void *obj)
{
  int index=findIndex(key,tableSize,keyTable);
  if ((index==-1)||((nElem+1)>(int)(loadFactor*tableSize)))
    {//The table is too small-- increase its size
      rehash(primeLargerThan(tableSize));
      //Find the key location in the new table
      index=findIndex(key,tableSize,keyTable);
    }
  DEBUGF(("Inserting new key at %d\n",index))
    void *oldObject=objTable[index];
  if (keyTable[index]==NULL)//This location had no key--
    keyTable[index]=key.newKey();
  //Plop the new object into the table
  objTable[index]=obj;
  nElem++;
  return oldObject;
}

//Look up the given object in this table.  Return NULL if not found.
void *Hashtable::get(const HashKey &key) const
{
  int index=findIndex(key,tableSize,keyTable);
  if (keyTable[index]!=NULL)
    return objTable[index];//The object is in the table
  else
    return NULL;//The key is not in the table
}

//Remove this object from the hashtable (re-hashing if needed)
// Returns the old object, if there was one.
void *Hashtable::remove(const HashKey &doomedKey)
{
  int index=findIndex(doomedKey,tableSize,keyTable);
  if ((index!=-1)&&(keyTable[index]!=NULL))
    {//The doomed key is in the table-- remove it
      DEBUGF(("Removing key at index %d\n",index))
	delete keyTable[index];
      void *oldValue=objTable[index];
      keyTable[index]=NULL;objTable[index]=NULL;
      
      //Now we have to re-hash all contiguous objects below--
      // these may have been bumped down because of the doomedKey.
      index=(index+1)%tableSize;
      while (keyTable[index]!=NULL) {
	//Here's a key that may need to be moved-- re-hash it.
	HashKey *key=keyTable[index];void *obj=objTable[index];
	keyTable[index]=NULL;objTable[index]=NULL;
	int newIndex=findIndex(*key,tableSize,keyTable);
	keyTable[newIndex]=key;objTable[newIndex]=obj;
	DEBUGF(("Removal re-hash: %d -> %d\n",index,newIndex)) 
	  //Advance to next location
	  index++;if (index>=tableSize) index=0;//Wrap around to start
      }
      return oldValue;//Return old object to user
    } 
  else return NULL;//Key wasn't in hash table
}

//Return an iterator for the objects in this hash table
HashtableIterator *Hashtable::objects(void)
{
  return new HashtableIterator(keyTable,objTable,tableSize);
}

//////////////////////// HashtableIterator //////////////////
//A HashtableIterator lets you easily list all the objects
// in a hash table (without knowing all the keys).

HashtableIterator::HashtableIterator(HashKeyPtr *NkeyTable,
				     HashObjectPtr *NobjTable,
				     int nLen)
{
  keyTable=NkeyTable;
  objTable=NobjTable;
  tableLen=nLen;
  curNo=0;
}

//Seek to start of hash table
void HashtableIterator::seekStart(void) {curNo=0;}

//Seek forward (or back) n hash slots
void HashtableIterator::seek(int n)
{
  curNo+=n;
  if (curNo<0) curNo=0;
  if (curNo>tableLen) curNo=tableLen;
}
	
//Return 1 if next will be non-NULL
int HashtableIterator::hasNext(void)
{
  while (curNo<tableLen) {
    if (keyTable[curNo]!=NULL)
      return 1;//We have a next object
    else 
      curNo++;//This spot is blank-- skip over it
  }
  return 0;//We went through the whole table-- no object
}

//Return the next object, or NULL if none
// The corresponding object key will be returned in retKey.
void *HashtableIterator::next(HashKey **retKey)
{
  while (curNo<tableLen) {
    if (keyTable[curNo]!=NULL) {
      //Here's the next object
      if (retKey) *retKey=keyTable[curNo];
      return objTable[curNo++];
    } else 
      curNo++;//This spot is blank-- skip over it
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
