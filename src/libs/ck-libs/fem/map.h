/*
 Some definitions of structures/classes used in map.C and possibly other FEM source files.
 Moved to this file by Isaac Dooley 4/5/05

 FIXME: Clean up the organization of this file, and possibly move other structures from Map.C here
        Also all the stuff in this file might just belong in fem_impl.h. I just moved it here so it could
	be included in fem.C for my element->element build adjacency table function.
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "fem_impl.h"
#include "fem.h"
#include "fem_mesh.h"
#include "cktimer.h"

#ifndef __FEM_MAP_H___
#define __FEM_MAP_H___

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


#endif // end of map.h header file
