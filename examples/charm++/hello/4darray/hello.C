/** \file hello.C
 *  Author: Abhinav S Bhatele
 *  Date Created: November 4th, 2007
 */

#include "hello.decl.h"
#include <stdio.h>
#include "ckmulticast.h"
#include "TopoManager.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int numW;
/*readonly*/ int numX;
/*readonly*/ int numY;
/*readonly*/ int numZ;

/** \class Main
 *
 */
class Main : public CBase_Main
{
public:
  CProxy_Hello arr;
  CProxySection_Hello secProxy;
  int numElems;
  int counter;

  Main(CkArgMsg* m)
  {
    numElems = counter = 0;
    //Process command-line arguments
    numW = numX = numY = numZ = 5;
    if(m->argc > 1) {
      if(m->argc != 5)
	CkPrintf("argc %d, 4 arguments needed, one for each dimension\n",m->argc);
      else {
	numW = atoi(m->argv[1]);
	numX = atoi(m->argv[2]);
	numY = atoi(m->argv[3]);
	numZ = atoi(m->argv[4]);
      }
    }
    delete m;

    //Start the computation
    CkPrintf("Running Hello on %d processors for [%d][%d][%d][%d] elements\n",
	     CkNumPes(), numW, numX, numY, numZ);
    mainProxy = thisProxy;

    CProxy_HelloMap map = CProxy_HelloMap::ckNew(numW, numX, numY, numZ);
    CkArrayOptions opts;
    opts.setMap(map);
    arr = CProxy_Hello::ckNew(opts);

    for(int i1=0; i1<numW; i1++)
      for(int i2=0; i2<numX; i2++)
	for(int i3=0; i3<numY; i3++)
	  for(int i4=0; i4<numZ; i4++) {
	    arr(i1, i2, i3, i4).insert();
	  }
    arr.doneInserting();
    CkPrintf("Array created\n");


    CkVec<CkArrayIndex4D> elems;    // add array indices
    for (short int i=0; i<numW; i+=2)
      for (short int j=0; j<numX; j+=3)
	for (short int k=0; k<numY; k+=4)
	  for (short int l=0; l<numZ; l+=5)
	    elems.push_back(CkArrayIndex4D(i % numW, j % numX, k % numY, l % numZ)); 
    numElems = elems.size();
    secProxy = CProxySection_Hello::ckNew(arr, elems.getVec(), numElems);

    CkPrintf("Section created\n");

    arr(0, 0, 0, 0).SayHi(17);
  };

  void done_1(void)
  {
    CkPrintf("Phase 1 done\n");
    secProxy.SayBye();
    // CkExit();
  };

  void done_2(void)
  {
    counter++;
    if(counter == numElems) {
      CkPrintf("Phase 2 done\n");
      CkPrintf("All done\n");
      CkExit();
    }
  };
};

/** \class Hello
 *
 */
class Hello : public CBase_Hello 
{
public:
  Hello()
  {
    //CkPrintf("Hello %d %d %d %d created\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z);
  }

  Hello(CkMigrateMessage *m) {}
  
  void SayHi(int hiNo)
  {
    char *name=(char *)malloc(sizeof(char)*1000);
    CkPrintf("Hi [%d] from element %d %d %d %d\n", hiNo, thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z);
    //Pass the hello on:
    if(thisIndex.w < numW-1)
      thisProxy(thisIndex.w+1, thisIndex.x, thisIndex.y, thisIndex.z).SayHi(hiNo+1);
    else if(thisIndex.x < numX-1)
      thisProxy(thisIndex.w, thisIndex.x+1, thisIndex.y, thisIndex.z).SayHi(hiNo+1);
    else if(thisIndex.y < numY-1)
      thisProxy(thisIndex.w, thisIndex.x, thisIndex.y+1, thisIndex.z).SayHi(hiNo+1);
    else if(thisIndex.z < numZ-1)
      thisProxy(thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z+1).SayHi(hiNo+1);
    else 
      //We've been around once-- we're done.
      mainProxy.done_1();
  }

  void SayBye()
  {
    CkPrintf("Bye from element %d %d %d %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z);
    mainProxy.done_2();
  }
};

/** \class HelloMap
 *
 */
class HelloMap : public CkArrayMap {
  public:
    int ****mapping;

    HelloMap(int w, int x, int y, int z) {
      int i, j, k;
      mapping = new int***[w];
      for (i=0; i<w; i++) {
        mapping[i] = new int**[x];
	for(j=0; j<x; j++) {
	  mapping[i][j] = new int*[y];
	    for(k=0; k<y; k++)
	      mapping[i][j][k] = new int[z]; 
	}
      }
      /* naively fold onto the 1d rank array with z innermost */
      for(int i=0; i<w; i++)
	for(int j=0; j<x; j++)
	  for(int k=0; k<y; k++)
	    for(int l=0; l<z; l++) {
	      mapping[i][j][k][l] = (i*x*y*z+ j*y*z + k*z +l)%CkNumPes();
	    }

    }

    int procNum(int, const CkArrayIndex &idx) {
      short *index = (short *)idx.data();
      return mapping[index[0]][index[1]][index[2]][index[3]]; 
    }
};

#include "hello.def.h"
