#include <stdio.h>
#include "charm++.h"

//*MUST* declare index type *BEFORE* we include hello.decl.h!

//This is our application-specific index type.  It's a
//  completely ordinary C++ class (or even C struct)-- the
//  only requirement is that all the data be allocated
//  locally (no pointers, no virtual methods).
class Fancy {
	int a,b;
public:
	Fancy() :a(0), b(0) {}
	Fancy(int a_,int b_) :a(a_), b(b_) {}
	Fancy next(void) const 
		{return Fancy(b-a,b+1);}
	int cardinality(void) const 
		{return b;}
	int getA(void) const {return a;}
	int getB(void) const {return b;}	
};

//This adapts the application's index for use by the array
// manager.  This class is only used by the translator--
// you never need to refer to it again!
class CkArrayIndexFancy : public CkArrayIndex {
	Fancy *idx;
public:
    CkArrayIndexFancy() 
    {
        /// Use placement new to ensure that the custom index object is placed in the memory reserved for it in the base class
        idx = new(index) Fancy(); 
    }

	CkArrayIndexFancy(const Fancy &f)
	{
        /// Use placement new to ensure that the custom index object is placed in the memory reserved for it in the base class
        idx = new(index) Fancy(f);
		nInts=2;
	}
};

#include "hello.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int nElements;

/*mainchare*/
class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    //Process command-line arguments
    nElements=5;
    if(m->argc >1 ) nElements=atoi(m->argv[1]);
    delete m;

    //Start the computation
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    mainProxy = thisProxy;

    CProxy_Hello arr = CProxy_Hello::ckNew();
    const Fancy startIndex(23,0);
    Fancy f=startIndex;
    for (int i=0;i<nElements;i++) {
      arr[f].insert();
      f=f.next();
    }
    arr.doneInserting();

    arr[startIndex].SayHi(17);
  };

  void done(void)
  {
    CkPrintf("All done\n");
    CkExit();
  };
};

/*array [Fancy]*/
class Hello : public CBase_Hello
{
public:
  Hello()
  {
    //Note how thisIndex is of type fancyIndex:
    CkPrintf("Hello (%d,%d) created on %d\n",
	thisIndex.getA(),thisIndex.getB(),CkMyPe());
  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo)
  {
    CkPrintf("Hi[%d] from element %d\n",hiNo,thisIndex.cardinality());
    if (thisIndex.cardinality() < nElements-1)
      //Pass the hello on:
      thisProxy[thisIndex.next()].SayHi(hiNo+1);
    else 
      //We've been around once-- we're done.
      mainProxy.done();
  }
};

#include "hello.def.h"
