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
which will perform all needed packing/unpacking.

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

#include <stdio.h> //<- for FILE *
#if 0
#  include <converse.h> // <- for CmiBool
#else
#  include <conv-mach.h>
#  define CmiBool bool
#  define CmiTrue true
#  define CmiFalse false
#endif


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


//The abstract base class:  PUP::er.
class er {
 private:
  er(const er &p);//You don't want to copy PUP::er's.
 protected:
  er() {}//You don't want to create raw PUP::er's.
 protected:
  //Generic bottleneck: pack/unpack n items of size itemSize 
  // and data type t from p.  Desc describes the data item
  virtual void bytes(void *p,int n,size_t itemSize,dataType t,const char *desc) =0;

 public:
  virtual ~er();//<- does nothing, but might be needed by some child
  
  //State queries (exactly one of these will be true)
  virtual CmiBool isSizing(void) const;
  virtual CmiBool isPacking(void) const;//<- these all default to false
  virtual CmiBool isUnpacking(void) const;
  virtual void *getBuf(int n) { return 0; }

//For single elements, pretend it's an array containing one element
  void operator()(signed char &v,const char *desc=NULL)     {(*this)(&v,1,desc);}
  void operator()(char &v,const char *desc=NULL)            {(*this)(&v,1,desc);}
  void operator()(short &v,const char *desc=NULL)           {(*this)(&v,1,desc);}
  void operator()(int &v,const char *desc=NULL)             {(*this)(&v,1,desc);}
  void operator()(long &v,const char *desc=NULL)            {(*this)(&v,1,desc);}
  void operator()(unsigned char &v,const char *desc=NULL)   {(*this)(&v,1,desc);}
  void operator()(unsigned short &v,const char *desc=NULL)  {(*this)(&v,1,desc);}
  void operator()(unsigned int &v,const char *desc=NULL)    {(*this)(&v,1,desc);}
  void operator()(unsigned long &v,const char *desc=NULL)   {(*this)(&v,1,desc);}
  void operator()(float &v,const char *desc=NULL)           {(*this)(&v,1,desc);}
  void operator()(double &v,const char *desc=NULL)          {(*this)(&v,1,desc);}
  void operator()(CmiBool &v,const char *desc=NULL)            {(*this)(&v,1,desc);}

//For arrays:
  //Integral types:
  void operator()(signed char *a,int nItems,const char *desc=NULL) 
    {bytes((void *)a,nItems,sizeof(signed char),Tchar,desc);}
  void operator()(char *a,int nItems,const char *desc=NULL) 
    {bytes((void *)a,nItems,sizeof(char),Tchar,desc);}
  void operator()(short *a,int nItems,const char *desc=NULL) 
    {bytes((void *)a,nItems,sizeof(short),Tshort,desc);}
  void operator()(int *a,int nItems,const char *desc=NULL)
    {bytes((void *)a,nItems,sizeof(int),Tint,desc);}
  void operator()(long *a,int nItems,const char *desc=NULL)
    {bytes((void *)a,nItems,sizeof(long),Tlong,desc);}
  
  //Unsigned integral types:
  void operator()(unsigned char *a,int nItems,const char *desc=NULL) 
    {bytes((void *)a,nItems,sizeof(unsigned char),Tuchar,desc);}
  void operator()(unsigned short *a,int nItems,const char *desc=NULL)
    {bytes((void *)a,nItems,sizeof(unsigned short),Tushort,desc);}
  void operator()(unsigned int *a,int nItems,const char *desc=NULL)
    {bytes((void *)a,nItems,sizeof(unsigned int),Tuint,desc);}
  void operator()(unsigned long *a,int nItems,const char *desc=NULL)
    {bytes((void *)a,nItems,sizeof(unsigned long),Tulong,desc);}
  
  //Floating-point types:
  void operator()(float *a,int nItems,const char *desc=NULL)
    {bytes((void *)a,nItems,sizeof(float),Tfloat,desc);}
  void operator()(double *a,int nItems,const char *desc=NULL)
    {bytes((void *)a,nItems,sizeof(double),Tdouble,desc);}
  
  //For bools:
  void operator()(CmiBool *a,int nItems,const char *desc=NULL)
    {bytes((void *)a,nItems,sizeof(CmiBool),Tbool,desc);}
  
  //For raw memory (n gives number of bytes)
  void operator()(void *a,int nBytes,const char *desc=NULL)
    {bytes((void *)a,nBytes,1,Tbyte,desc);}
};

//Superclass of packers
class packer : public er {
 public:
  virtual CmiBool isPacking(void) const;
};

//Superclass of unpackers
class unpacker : public er {
 public://<- for some reason the xlator needs "bytes" to be public
  //Generic bottleneck: pack/unpack n items of size itemSize 
  // and data type t from p.  Desc describes the data item
  virtual void bytes(void *p,int n,size_t itemSize,dataType t,const char *desc) =0;
 public:
  virtual CmiBool isUnpacking(void) const;
};

//For finding the number of bytes to pack (e.g., to preallocate a memory buffer)
class sizer : public er {
 protected:
  int nBytes;
  //Generic bottleneck: n items of size itemSize
  virtual void bytes(void *p,int n,size_t itemSize,dataType t,const char *desc);
 public:
  //Write data to the given buffer
  sizer(void) {nBytes=0;}
  virtual CmiBool isSizing(void) const;
  
  //Return the current number of bytes to be packed
  int size(void) const {return nBytes;}
};

//For packing into a preallocated, presized memory buffer
class toMem : public packer {
 protected:
  myByte *buf;//destination buffer (stuff gets packed *in* here)
  //Generic bottleneck: pack n items of size itemSize from p.
  virtual void bytes(void *p,int n,size_t itemSize,dataType t,const char *desc);
 public:
  //Write data to the given buffer
  toMem(void *Nbuf) {buf=(myByte *)Nbuf;}
  virtual void *getBuf(int n)
  {
    void *ret = (void*) buf; 
    buf += n; 
    return ret;
  }
};

//For unpacking from a memory buffer
class fromMem : public unpacker {
 protected:
  const myByte *buf;//source buffer (stuff gets unpacked *from* here)
  //Generic bottleneck: unpack n items of size itemSize from p.
  virtual void bytes(void *p,int n,size_t itemSize,dataType t,const char *desc);
 public:
  //Read data from the given buffer
  fromMem(const void *Nbuf) {buf=(const myByte *)Nbuf;}
};

//For packing to a disk file
class toDisk : public packer {
 protected:
  FILE *outF;
  //Generic bottleneck: pack n items of size itemSize from p.
  virtual void bytes(void *p,int n,size_t itemSize,dataType t,const char *desc);
 public:
  //Write data to the given file pointer 
  // (must be opened for binary write)
  toDisk(FILE *f) {outF=f;}
};

//For unpacking from a disk file
class fromDisk : public unpacker {
 protected:
  FILE *inF;
  //Generic bottleneck: unpack n items of size itemSize from p.
  virtual void bytes(void *p,int n,size_t itemSize,dataType t,const char *desc);
 public:
  //Write data to the given file pointer 
  // (must be opened for binary read)
  fromDisk(FILE *f) {inF=f;}
};



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
class xlater : public unpacker {
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
  
  unpacker &myUnpacker;//Raw data unpacker
  //Generic bottleneck: unpack n items of size itemSize from p.
  virtual void bytes(void *p,int n,size_t itemSize,dataType t,const char *desc);
 public:
  xlater(const machineInfo &fromMachine, unpacker &fromData);
};

};//<- End "namespace" PUP

//This catches "p|t"'s for user-defined types T:
template <class T>
void operator|(PUP::er &p,T &t)
{
         p((void *)&t,sizeof(T));
}






