/*
Pack/UnPack Library for UIUC Parallel Programming Lab
Orion Sky Lawlor, olawlor@uiuc.edu, 4/5/2000

This library allows you to easily pack an array, structure,
or object into a memory buffer or disk file, and then read 
the object back later.  The library can also handle translating
between different machine representations for integers and floats.

Typically, the user has to write separate functions for buffer 
sizing, pack to memory, unpack from memory, pack to disk, and 
unpack from disk.  These functions all perform the exact same function--
namely, they list the members of the array, struct, or object.
Further, all the functions must agree, or the unpacked data will 
be garbage.  This library allows the user to write *one* function,
pup, which will perform all needed packing/unpacking.

A simple example is:
class foo {
 private:
  CmiBool isBar;
  int x;
  char y;
  unsigned long z;
  float q[3];
 public:
  ...
  void pup(PUP::er &p) {
    p(isBar);
    p(x);p(y);p(z);
    p(q,3);
  }
};

A more complex example is:
class bar {
 private:
  foo f;
  int nArr;//Length of array below
  double *arr;//Heap-allocated array
 public:
  ...
  
  void pup(PUP::er &p) {
    f.pup(p);
    p(nArr);
    if (p.isUnpacking())
      arr=new double[nArr];
    p(arr,nArr);
  }
};
*/

#ifndef __CK_PUP_H
#define __CK_PUP_H

#include <stdio.h> /*<- for "FILE *" */

#ifndef __cplusplus
#error "Use pup_c.h for C programs-- pup.h is for C++ programs"
#endif

#if 0
#  include <converse.h> // <- for CmiBool
#else
#  include <conv-mach.h>
#  include "conv-autoconfig.h"
// you cannot define Bool twice !!
#ifndef CONVERSE_H
#if CMK_BOOL_UNDEFINED
enum CmiBool {CmiFalse=0, CmiTrue=1};
#else
typedef bool CmiBool;
#define CmiFalse false
#define CmiTrue true
#endif
#endif
/*
#  define CmiBool bool
#  define CmiTrue true
#  define CmiFalse false
*/
#endif


//We need CkMigrateMessage only to distinguish the migration
// constructor from all other constructors-- the type
// itself has no meaningful fields.
typedef struct {int is_only_a_name;} CkMigrateMessage;

class PUP {//<- Should be "namespace", once all compilers support them
 public:
 
//Item data types-- these are used to do byte swapping, etc.
typedef enum {
//(this list must exactly match that in PUPer_xlate)
  Tchar=0,Tshort, Tint, Tlong,
  Tuchar,Tushort,Tuint,Tulong,
  Tfloat,Tdouble,
  Tbool,
  Tbyte,
  dataType_last //<- for setting table lengths, etc.
} dataType;

//This should be a 1-byte unsigned type
typedef unsigned char myByte;

//Forward declarations
class er;
class able;
class xlater;

//Used for out-of-order unpacking
class seekBlock {
	enum {maxSections=3};
	int secTab[maxSections+1];//The start of each seek section
	int nSec;//Number of sections; current section #
	int secTabOff;//Start of the section table, relative to the seek block
	er &p;
	CmiBool hasEnded;
public:
	//Constructor
	seekBlock(er &Np,int nSections);
	//Destructor
	~seekBlock();

	//Seek to the given section number (0-based, less than nSections)
	void seek(int toSection);
	//Finish with this seeker (must be called)
	void endBlock(void);
	
	//An evil hack to avoid inheritance and virtual functions among seekers--
	// stores the PUP::er specific block start information.
	union {
		int off;
		long loff;
		const myByte *cptr;
		myByte *ptr;
		void *vptr;
	} data;
};

//The abstract base class:  PUP::er.
class er {
 private:
  er(const er &p);//You don't want to copy PUP::er's.
 protected:
  enum {IS_DELETING =0x0008};
  enum {IS_SIZING   =0x0100,
  	IS_PACKING  =0x0200,
        IS_UNPACKING=0x0400,
        TYPE_MASK   =0xFF00};
  unsigned int PUP_er_state;
#if CMK_EXPLICIT
  explicit er(unsigned int inType) //You don't want to create raw PUP::er's.
#else
  er(unsigned int inType) //You don't want to create raw PUP::er's.
#endif
    {PUP_er_state=inType;}
 public:
  virtual ~er();//<- does nothing, but might be needed by some child
  
  //State queries (exactly one of these will be true)
  CmiBool isSizing(void) const {return (PUP_er_state&IS_SIZING)!=0?CmiTrue:CmiFalse;}
  CmiBool isPacking(void) const {return (PUP_er_state&IS_PACKING)!=0?CmiTrue:CmiFalse;}
  CmiBool isUnpacking(void) const {return (PUP_er_state&IS_UNPACKING)!=0?CmiTrue:CmiFalse;}

  //This indicates that the pup routine should free memory during packing.
  void becomeDeleting(void) {PUP_er_state|=IS_DELETING;}
  CmiBool isDeleting(void) const {return (PUP_er_state&IS_DELETING)!=0?CmiTrue:CmiFalse;}
  
//For single elements, pretend it's an array containing one element
  void operator()(signed char &v)     {(*this)(&v,1);}
#if CMK_SIGNEDCHAR_DIFF_CHAR
  void operator()(char &v)            {(*this)(&v,1);}
#endif
  void operator()(short &v)           {(*this)(&v,1);}
  void operator()(int &v)             {(*this)(&v,1);}
  void operator()(long &v)            {(*this)(&v,1);}
  void operator()(unsigned char &v)   {(*this)(&v,1);}
  void operator()(unsigned short &v)  {(*this)(&v,1);}
  void operator()(unsigned int &v)    {(*this)(&v,1);}
  void operator()(unsigned long &v)   {(*this)(&v,1);}
  void operator()(float &v)           {(*this)(&v,1);}
  void operator()(double &v)          {(*this)(&v,1);}
  void operator()(CmiBool &v)            {(*this)(&v,1);}

//For arrays:
  //Integral types:
  void operator()(signed char *a,int nItems) 
    {bytes((void *)a,nItems,sizeof(signed char),Tchar);}
#if CMK_SIGNEDCHAR_DIFF_CHAR
  void operator()(char *a,int nItems) 
    {bytes((void *)a,nItems,sizeof(char),Tchar);}
#endif
  void operator()(short *a,int nItems) 
    {bytes((void *)a,nItems,sizeof(short),Tshort);}
  void operator()(int *a,int nItems)
    {bytes((void *)a,nItems,sizeof(int),Tint);}
  void operator()(long *a,int nItems)
    {bytes((void *)a,nItems,sizeof(long),Tlong);}
  
  //Unsigned integral types:
  void operator()(unsigned char *a,int nItems) 
    {bytes((void *)a,nItems,sizeof(unsigned char),Tuchar);}
  void operator()(unsigned short *a,int nItems)
    {bytes((void *)a,nItems,sizeof(unsigned short),Tushort);}
  void operator()(unsigned int *a,int nItems)
    {bytes((void *)a,nItems,sizeof(unsigned int),Tuint);}
  void operator()(unsigned long *a,int nItems)
    {bytes((void *)a,nItems,sizeof(unsigned long),Tulong);}
  
  //Floating-point types:
  void operator()(float *a,int nItems)
    {bytes((void *)a,nItems,sizeof(float),Tfloat);}
  void operator()(double *a,int nItems)
    {bytes((void *)a,nItems,sizeof(double),Tdouble);}
  
  //For bools:
  void operator()(CmiBool *a,int nItems)
    {bytes((void *)a,nItems,sizeof(CmiBool),Tbool);}
  
  //For raw memory (n gives number of bytes)
  void operator()(void *a,int nBytes)
    {bytes((void *)a,nBytes,1,Tbyte);}
  
  //For allocatable objects (system will new/delete object and call pup routine)
  void operator()(able** a)
    {object(a);}
  //For pre- or stack-allocated PUP::able objects-- just call object's pup
  void operator()(able& a)
    {a.pup(*this);}
 
 protected:
  //Generic bottleneck: pack/unpack n items of size itemSize 
  // and data type t from p.  Desc describes the data item
  friend class xlater;
  virtual void bytes(void *p,int n,size_t itemSize,dataType t) =0;
  virtual void object(able** a);
  
  //For seeking (pack/unpack in different orders)
  friend class seekBlock;
  virtual void impl_startSeek(seekBlock &s); /*Begin a seeking block*/
  virtual int impl_tell(seekBlock &s); /*Give the current offset*/
  virtual void impl_seek(seekBlock &s,int off); /*Seek to the given offset*/
  virtual void impl_endSeek(seekBlock &s);/*End a seeking block*/
};

/************** PUP::er -- Sizer ******************/
//For finding the number of bytes to pack (e.g., to preallocate a memory buffer)
class sizer : public er {
 protected:
  int nBytes;
  //Generic bottleneck: n items of size itemSize
  virtual void bytes(void *p,int n,size_t itemSize,dataType t);
 public:
  //Write data to the given buffer
  sizer(void):er(IS_SIZING) {nBytes=0;}
  
  //Return the current number of bytes to be packed
  int size(void) const {return nBytes;}
};

/********** PUP::er -- Binary memory buffer pack/unpack *********/
class mem : public er { //Memory-buffer packers and unpackers
 protected:
  myByte *origBuf;//Start of memory buffer
  myByte *buf;//Memory buffer (stuff gets packed into/out of here)
  mem(unsigned int type,myByte *Nbuf):er(type),origBuf(Nbuf),buf(Nbuf) {}

  //For seeking (pack/unpack in different orders)
  virtual void impl_startSeek(seekBlock &s); /*Begin a seeking block*/
  virtual int impl_tell(seekBlock &s); /*Give the current offset*/
  virtual void impl_seek(seekBlock &s,int off); /*Seek to the given offset*/
 public:
  //Return the current number of buffer bytes used
  int size(void) const {return buf-origBuf;}
};

//For packing into a preallocated, presized memory buffer
class toMem : public mem {
 protected:
  //Generic bottleneck: pack n items of size itemSize from p.
  virtual void bytes(void *p,int n,size_t itemSize,dataType t);
 public:
  //Write data to the given buffer
  toMem(void *Nbuf):mem(IS_PACKING,(myByte *)Nbuf) {}
};

//For unpacking from a memory buffer
class fromMem : public mem {
 protected:
  //Generic bottleneck: unpack n items of size itemSize from p.
  virtual void bytes(void *p,int n,size_t itemSize,dataType t);
 public:
  //Read data from the given buffer
  fromMem(const void *Nbuf):mem(IS_UNPACKING,(myByte *)Nbuf) {}
};

/********** PUP::er -- Binary disk file pack/unpack *********/
class disk : public er {
 protected:
  FILE *F;//Disk file to read from/write to
  disk(unsigned int type,FILE *f):er(type),F(f) {}
  virtual ~disk();

  //For seeking (pack/unpack in different orders)
  virtual void impl_startSeek(seekBlock &s); /*Begin a seeking block*/
  virtual int impl_tell(seekBlock &s); /*Give the current offset*/
  virtual void impl_seek(seekBlock &s,int off); /*Seek to the given offset*/
};

//For packing to a disk file
class toDisk : public disk {
 protected:
  //Generic bottleneck: pack n items of size itemSize from p.
  virtual void bytes(void *p,int n,size_t itemSize,dataType t);
 public:
  //Write data to the given file pointer 
  // (must be opened for binary write)
  toDisk(FILE *f):disk(IS_PACKING,f) {}
};

//For unpacking from a disk file
class fromDisk : public disk {
 protected:
  //Generic bottleneck: unpack n items of size itemSize from p.
  virtual void bytes(void *p,int n,size_t itemSize,dataType t);
 public:
  //Write data to the given file pointer 
  // (must be opened for binary read)
  fromDisk(FILE *f):disk(IS_UNPACKING,f) {}
};

/********** PUP::er -- Heterogenous machine pack/unpack *********/
//This object describes the data representation of a machine.
class machineInfo {
 public:
  typedef unsigned char myByte;
  myByte magic[4];//Magic number (to identify machineInfo structs)
  myByte version;//0-- current version
  
  myByte intBytes[4]; //<- sizeof(char,short,int,long)
  myByte intFormat;//0-- big endian.  1-- little endian.
  
  myByte floatBytes; //<- sizeof(...)
  myByte doubleBytes;
  myByte floatFormat;//0-- big endian IEEE.  1-- little endian IEEE.
    
  myByte boolBytes;
  
  myByte padding[2];//Padding to 16 bytes

  //Return true if our magic number is valid.
  CmiBool valid(void) const;
  //Return true if we differ from the current (running) machine.
  CmiBool needsConversion(void) const;
  
  //Get a machineInfo for the current machine
  static const machineInfo &current(void);
};

//For translating some odd disk/memory representation into the 
// current machine representation.  (We really only need to
// translate during unpack-- "reader makes right".)
class xlater : public er {
 protected:
  typedef void (*dataConverterFn)(int N,const myByte *in,myByte *out,int nElem);
  
  //This table is indexed by dataType, and contains an appropriate
  // conversion function to unpack a n-item array of the corresponding 
  // data type (possibly in-place).
  dataConverterFn convertFn[dataType_last];
  //Maps dataType to source machine's dataSize
  size_t convertSize[dataType_last];
  void setConverterInt(const machineInfo &m,const machineInfo &cur,
    int isUnsigned,int intType,dataType dest);
  
  er &myUnpacker;//Raw data unpacker
  //Generic bottleneck: unpack n items of size itemSize from p.
  virtual void bytes(void *p,int n,size_t itemSize,dataType t);
 public:
  xlater(const machineInfo &fromMachine, er &fromData);
 protected:
  //For seeking (pack/unpack in different orders)
  friend class seekBlock;
  virtual void impl_startSeek(seekBlock &s); /*Begin a seeking block*/
  virtual int impl_tell(seekBlock &s); /*Give the current offset*/
  virtual void impl_seek(seekBlock &s,int off); /*Seek to the given offset*/
  virtual void impl_endSeek(seekBlock &s);/*End a seeking block*/
};

/*************** PUP::able support ***************/
//The base class of system-allocatable objects with pup routines
class able {
public:
	//A globally-unique, persistent identifier for an allocatable object
	class PUP_ID {
	public:
		enum {len=8};
		unsigned char hash[len];
		PUP_ID() {}
		PUP_ID(int val) {for (int i=0;i<len;i++) hash[i]=val;}
		PUP_ID(const char *name) {setName(name);}
		void setName(const char *name);//Write name into hash
		CmiBool operator==(const PUP_ID &other) const {
			for (int i=0;i<len;i++) 
				if (hash[i]!=other.hash[i])
					return CmiFalse;
			return CmiTrue;
		}
		void pup(er &p) {
			 p((void *)hash,sizeof(unsigned char)*len);
		}
		void pup(er &p) const {
			 p((void *)hash,sizeof(unsigned char)*len);
		}
	};

protected:
	able() {}
	able(CkMigrateMessage *) {}
	virtual ~able();//Virtual destructor may be needed by some child

public:
//Constructor function registration:
	typedef able* (*constructor_function)(void);
	static PUP_ID register_constructor(const char *className,
		constructor_function fn);
	static constructor_function get_constructor(const PUP_ID &id);

//Target methods:
	virtual void pup(er &p);
	virtual const PUP_ID &get_PUP_ID(void) const=0;

    friend inline void operator|(er &p,able &a) {a.pup(p);}
    friend inline void operator|(er &p,able* &a) {p(&a);}
};

//Declarations to include in a PUP::able's body
#define PUPable_decl(className) \
private: \
    static PUP::able *call_PUP_constructor(void); \
    static PUP::able::PUP_ID my_PUP_ID;\
public:\
    virtual const PUP::able::PUP_ID &get_PUP_ID(void) const; \
    static void register_PUP_ID(void); \
    friend inline void operator|(PUP::er &p,className &a) {a.pup(p);}\
    friend inline void operator|(PUP::er &p,className* &a) {\
	PUP::able *pa=a;  p(&pa);  a=(className *)pa;\
    }

//Definitions to include exactly once at file scope
#define PUPable_def(className) \
	PUP::able *className##::call_PUP_constructor(void) \
		{ return new className((CkMigrateMessage *)0);}\
	const PUP::able::PUP_ID &className##::get_PUP_ID(void) const\
		{ return className##::my_PUP_ID; }\
	PUP::able::PUP_ID className##::my_PUP_ID;\
	void className##::register_PUP_ID(void)\
		{my_PUP_ID=register_constructor(#className,\
		              className##::call_PUP_constructor);}\

//Code to execute exactly once at program start time
#define PUPable_reg(className) \
    className##::register_PUP_ID();

};//<- End "namespace" PUP

/******** PUP via pipe: another way to access PUP::ers ******/

//This catches "p|t" for all user-defined types T:
template <class T>
inline void operator|(PUP::er &p,T &t)
{
         p((void *)&t,sizeof(T));
}

//These more specific versions map p|t to p(t) for all handled types
inline void operator|(PUP::er &p,signed char &t) {p(t);}
#if CMK_SIGNEDCHAR_DIFF_CHAR
inline void operator|(PUP::er &p,char &t) {p(t);}
#endif
inline void operator|(PUP::er &p,unsigned char &t) {p(t);}
inline void operator|(PUP::er &p,short &t) {p(t);}
inline void operator|(PUP::er &p,int &t) {p(t);}
inline void operator|(PUP::er &p,long &t) {p(t);}
inline void operator|(PUP::er &p,unsigned short &t) {p(t);}
inline void operator|(PUP::er &p,unsigned int &t) {p(t);}
inline void operator|(PUP::er &p,unsigned long &t) {p(t);}
inline void operator|(PUP::er &p,float &t) {p(t);}
inline void operator|(PUP::er &p,double &t) {p(t);}
inline void operator|(PUP::er &p,CmiBool &t) {p(t);}

#define PUPmarshall(type) \
  inline void operator|(PUP::er &p,type &t) {t.pup(p);}
#define PUPmarshal(type) PUPmarshall(type) /*Support this common misspelling*/

#endif //def __CK_PUP_H


