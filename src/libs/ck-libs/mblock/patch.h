/*
A patch is a portion of a block's face--
an interface between this block and something else.

Orion Sky Lawlor, olawlor@acm.org, 6/13/2001
*/
#ifndef __CSAR_PATCH_H
#define __CSAR_PATCH_H

#include <charm++.h>
#include "gridutil.h"

//A rectangular portion of a face 
class patch {
public:
	//Location in the source block:
	/*Beginning and ending node+1 for this patch (0-based, c-style)*/
	blockLoc start,end;
	enum {ext,send,recv} type; 
  virtual void print(void) = 0;
  void pup(PUP::er &p)
  {
    start.pup(p);
    end.pup(p);
    p((int&)type);
  }
  // return the width of this patch in w.
  // width is calculated in extremely ad-hoc way right now. 
  // It is the narrowest dimension of the patch. 
  // returns 0, 1, or 2
  int getWidth(blockLoc &w)
  {
    int dx = end[0] - start[0];
    int dy = end[1] - start[1];
    int dz = end[2] - start[2];
    int retval;
    w[0] = w[1] = w[2] = 0;
    if (dx < dy) {
      if(dx < dz) {
        // dx is minimum
        w[0] = dx; retval = 1;
      } else {
        // dz is minimum
        w[2] = dz;retval = 2;
      }
    } else {
      if(dy < dz) {
        // dy is minimum
        w[1] = dy;retval = 1;
      } else {
        // dz is minimum
        w[2] = dz;retval = 2;
      }
    }
  }
  // returns the position of the patch w.r.t the block in the dim dimension
  // dim = 0, 1, 2 for x, y,and z respectively
  // position = 1 or -1 depending on whether the patch lies near the start of
  // the block or the end of the block respectively
  int getPos(int dim)
  {
    return (start[dim] == 0) ? 1 : (-1);
  }

protected:
	patch(const blockLoc &start_,const blockLoc &end_)
	  :start(start_),end(end_) { }
  patch(void) {}
};

//A patch that faces the external world--
// between a block and "outside"
class externalBCpatch : public patch {
public:
	//Gridgen boundary condition number
	int bcNo;

  externalBCpatch(void) {}
	externalBCpatch(
	     const blockLoc &srcStart_,const blockLoc &srcEnd_,
	     int bcNo_)
	  : patch(srcStart_,srcEnd_), bcNo(bcNo_) { type=ext; }
  void pup(PUP::er &p)
  {
    patch::pup(p);
    p(bcNo);
  }
  void print(void)
  {
    CkPrintf("Type=external, start=%d,%d,%d end=%d,%d,%d bc#=%d\n",
        start[0],start[1],start[2],end[0],end[1],end[2],bcNo);
  }
};


//Permutation of (i,j,k) from source to dest:
//  Source axis i maps to dest. axis orient[i]. (zero-based)
class orientation {
  int orient[3];
  int flip[3];
 public:
  orientation(void) {}
  orientation(const int *codedOrient);
  int operator[](int axis) const {return orient[axis];}
  int isFlipped(int axis) const {return flip[axis];}
  // this is an inverse of operator[]
  int getMap(int axis)
  {
    for(int i=0;i<3;i++)
      if(axis == orient[i])
        return i;
    // error if comes here
  }
  void pup(PUP::er &p)
  {
    p(orient[0]); p(orient[1]); p(orient[2]);
    p(flip[0]); p(flip[1]); p(flip[2]);
  }
  void print(void)
  {
    CkPrintf("orient=%d,%d,%d flip=%d,%d,%d\n",
        orient[0],orient[1],orient[2],flip[0],flip[1],flip[2]);
  }
};

//An internal boundary, between blocks
class internalBCpatch : public patch {
public:
	int dest; //The destination block
	int destPatch; //(0-based) index of our partner on the dest. block
	orientation orient; //His orientation relative to us

  internalBCpatch(void) {}
	internalBCpatch(int dest_,int destPatch_,const int *codedOrient,
	     const blockLoc &srcStart_,const blockLoc &srcEnd_)
	  : patch(srcStart_,srcEnd_), 
	    dest(dest_),destPatch(destPatch_),
	    orient(codedOrient) { }
  void pup(PUP::er &p)
  {
    patch::pup(p);
    p(dest);
    p(destPatch);
    orient.pup(p);
  }
  void print(void)
  {
    CkPrintf("Type=internal, start=%d,%d,%d end=%d,%d,%d ",
        start[0],start[1],start[2],end[0],end[1],end[2]);
    CkPrintf("dest=%d[%d] ",dest, destPatch);
    orient.print();
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
    p((void*)locs, n*sizeof(vector3d));
    p(nPatches);
    if(p.isUnpacking())
      patches = new (patch*)[nPatches];
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
