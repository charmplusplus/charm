/*
A patch is a portion of a block's face--
an interface between this block and something else.

Orion Sky Lawlor, olawlor@acm.org, 6/13/2001
*/
#ifndef __CSAR_PATCH_H
#define __CSAR_PATCH_H

#include <charm++.h>
#include "gridutil.h"

//Describes how to extrude a patch into a 3D block
class extrudeMethod {
 public:
  int toWidth;//Number of cells to extrude
  bool withCorners;//If true, include diagonal cells
  extrudeMethod(int ghostWidth) {
	if (ghostWidth>=0) {
	   toWidth=ghostWidth; withCorners=false;
	} else {
	   toWidth=-ghostWidth; withCorners=true;
	}
  }
  extrudeMethod(int w,bool c) :toWidth(w),withCorners(c) { }
  extrudeMethod(void) { }
  void pup(PUP::er &p) {
    p(toWidth);
    p(withCorners);
  }
};

//A rectangular portion of a face 
class patch {
 protected:
	//Location in the source block:
	blockSpan span;

	int flatAxis; //Axis along which we're flat-- the normal axis
	bool isLow; //True if we're at zero along our flat axis
public:
	enum {ext=0,internal=1} type; 

	//Get the extents of this patch, extruded to toWidth
	blockSpan getExtents(const extrudeMethod &m,bool forVoxel,int dir=1);

	virtual void print(void) = 0;
	void pup(PUP::er &p)
	{
	  span.pup(p);
	  p((int&)type);
	  p(flatAxis);
	  p(isLow);
	}
	
	patch(void) { } /*Migration constructor*/
protected:
	patch(const blockSpan &span_);
};

//A patch that faces the external world--
// between a block and "outside"
class externalBCpatch : public patch {
public:
	//Gridgen boundary condition number
        int bcNo;
        externalBCpatch(void) {} /*Migration constructor*/
	externalBCpatch(
	     const blockSpan &srcSpan_,
	     int bcNo_)
	  : patch(srcSpan_), bcNo(bcNo_) { type=ext; }
  void pup(PUP::er &p)
  {
    patch::pup(p);
    p(bcNo);
  }
  void print(void)
  {
    CkPrintf("Type=external, bc#=%d, ",bcNo); span.print();
    CkPrintf("\n");
  }
};


//Permutation of (i,j,k) from source to dest:
//  Source axis i maps to dest. axis orient[i]. (zero-based)
class orientation {
  int s2d[3]; //Source to dest axis
  int flip[3]; //Source axis i maps to a flipped dest.
 public:
  orientation(void) {}
  orientation(const int *codedOrient);
  int operator[](int sAxis) const {return s2d[sAxis];}
  int dIsFlipped(int sAxis) const {return flip[sAxis];}
  
  //Map dest. axis to source axis
  int dToS(int dAxis) 
  {
    for(int sAxis=0;sAxis<3;sAxis++)
      if(dAxis == s2d[sAxis])
        return sAxis;
    // error if comes here
    return -1;
  }
  
  //Return the orientation, in source coordinates, of the given
  // destination axis.
  blockLoc destAxis(int dAxis) {
    int sAxis=dToS(dAxis);
    blockLoc ret(0,0,0);
    if (dIsFlipped(sAxis))
      ret[sAxis]=-1;
    else
      ret[sAxis]=1;
    return ret;
  }
  void pup(PUP::er &p)
  {
    p(s2d,3);
    p(flip,3);
  }
  void print(void)
  {
    CkPrintf(" s2d=%d,%d,%d flip=%d,%d,%d\n",
        s2d[0],s2d[1],s2d[2],flip[0],flip[1],flip[2]);
  }
};

//An internal boundary, between blocks
class internalBCpatch : public patch {
public:
	int dest; //The destination block
	int destPatch; //(0-based) index of our partner on the dest. block
	orientation orient; //His orientation relative to us

  internalBCpatch(void) {} /*Migration constructor*/
	internalBCpatch(int dest_,int destPatch_,const int *codedOrient,
	     const blockSpan &span_)
	  : patch(span_), 
	    dest(dest_),destPatch(destPatch_),
	    orient(codedOrient) { type=internal; }
  void pup(PUP::er &p)
  {
    patch::pup(p);
    p(dest);
    p(destPatch);
    orient.pup(p);
  }
  void print(void)
  {
    CkPrintf("Type=internal (%d:%d) ",dest,destPatch); 
    span.print();
    orient.print();
    CkPrintf("\n");
  }
};


//A simple "block" object
class block {
	blockDim dim;
	vector3d *locs;
	int nPatches;
	patch **patches;
public:
	block(const char *filePrefix,int blockNo);
  block(void) {};
	~block();
	
	const blockDim &getDim(void) const {return dim;}
	vector3d &getLoc(const blockLoc &i) {return locs[dim[i]];}
	const vector3d &getLoc(const blockLoc &i) const 
		{return locs[dim[i]];}
	
	int getPatches(void) const {return nPatches;}
	patch *getPatch(int i) const {return patches[i];}
  void print(void)
  {
    CkPrintf("dimensions: %d, %d, %d\n", dim[0], dim[1], dim[2]);
    CkPrintf("nPatches=%d\n", nPatches);
    for(int i=0;i<nPatches;i++) {
      CkPrintf("----patch %d----\n", i);
      patches[i]->print();
    }
  }
  void pup(PUP::er &p)
  {
    dim.pup(p);
    int n = dim.getSize();
    p(n);
    if(p.isUnpacking())
      locs = new vector3d[n];
    p((char*)locs, n*sizeof(vector3d));
    p(nPatches);
    if(p.isUnpacking())
      patches = new patch*[nPatches];
    for(int i=0;i<nPatches;i++) {
      if(p.isUnpacking()) {
        int type;
        p(type);
        if(type==patch::ext) {
          patches[i] = new externalBCpatch();
          ((externalBCpatch*)(patches[i]))->pup(p);
        } else {
          patches[i] = new internalBCpatch();
          ((internalBCpatch*)(patches[i]))->pup(p);
        }
      } else {
        int type = (int) patches[i]->type;
        p(type);
        if(type==patch::ext)
          ((externalBCpatch*)(patches[i]))->pup(p);
        else
          ((internalBCpatch*)(patches[i]))->pup(p);
      }
    }
    if (p.isPacking())
      delete this;
  }
};


#endif







