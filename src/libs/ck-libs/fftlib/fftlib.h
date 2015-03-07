#ifndef _fftlib_h_
#define _fftlib_h_

#include <charm++.h>
#include "ckcomplex.h"
#include "rfftw.h"

#define COMPLEX_TO_REAL -11
#define REAL_TO_COMPLEX -12
#define COMPLEX_TO_COMPLEX -13
#define NULL_TO_NULL -14

#define MAX_FFTS 5

class NormalFFTinfo {
 public:
    NormalFFTinfo(int sDim[2], int dDim[2], int isSrc,
		  void *dptr, int transType,
		  int sPlanesPerSlab=1, int dPlanesPerSlab=1) {
	init(sDim, dDim, isSrc, dptr, transType, sPlanesPerSlab, dPlanesPerSlab);
    }
    NormalFFTinfo(NormalFFTinfo &info) {
	init(info.srcSize, info.destSize, info.isSrcSlab,
	     info.dataPtr, info.transformType,
	     info.srcPlanesPerSlab, info.destPlanesPerSlab);
    }
    
    NormalFFTinfo(void) {dataPtr=NULL;}
    
    int srcSize[2], destSize[2];
    bool isSrcSlab;
    int srcPlanesPerSlab, destPlanesPerSlab;
    void *dataPtr;
    int transformType;
    
    void pup(PUP::er &p) {
	p(srcSize, 2);
	p(destSize, 2);
	p(isSrcSlab);
	p(srcPlanesPerSlab);
	p(destPlanesPerSlab);
	p|transformType;
	if (p.isUnpacking()) 
	    dataPtr = NULL;
    }

 private:  
    void init( int sDim[2], int dDim[2], int isSrc,
	       void *dptr, int transT,
	       int sPlanesPerSlab, int dPlanesPerSlab) {
	if (sDim[1] != dDim[1])
	    ckerr << "WARNING"
		  << "This configuration of the source and destination "
		  << "is not consistent, check the dimensions. The program is "
		  << "likely to misbehave"
		  << endl;
	isSrcSlab = isSrc;
	srcPlanesPerSlab = sPlanesPerSlab;
	destPlanesPerSlab = dPlanesPerSlab;
	dataPtr = dptr;
	transformType=transT;
	memcpy(srcSize, sDim, 2 * sizeof(int));
	memcpy(destSize, dDim, 2 * sizeof(int));
    }
};

typedef struct _PencilType{
    const static int XLINE = 0;
    const static int YLINE = 1;
    const static int ZLINE = 2;
}PencilType;

typedef struct _PencilBlock{
    const static int FLATBLOCK = 0;
    const static int SQUAREBLOCK = 1;
}PencilBlock;


class LineFFTinfo {
 public:
	// Constructors
	LineFFTinfo(int size[3], int _ptype, int _pblock, complex *dptr, int _xPencilsPerSlab=1, int _yPencilsPerSlab=1, int _zPencilsPerSlab=1) {
	  init(size[0], size[1], size[2], _ptype, _pblock, dptr, _xPencilsPerSlab, _yPencilsPerSlab, _zPencilsPerSlab);
	}
	LineFFTinfo(LineFFTinfo &info) {
	    init(info.sizeX, info.sizeY, info.sizeZ, info.ptype, info.pblock, (complex *) NULL, info.xPencilsPerSlab, info.yPencilsPerSlab, info.zPencilsPerSlab);
	}
	LineFFTinfo(void) {}

	// charm pup function
	void pup(PUP::er &p) {
	    p|sizeX;
	    p|sizeY;
	    p|sizeZ;
	    p(ptype);
	    p(pblock);
	    p(xPencilsPerSlab);
	    p(yPencilsPerSlab);
	    p(zPencilsPerSlab);
	    p(xsquare, 2);
	    p(ysquare, 2);
	    p(zsquare, 2);
	    if (p.isUnpacking()) 
		dataPtr = (complex *) NULL;
	}
	int sizeX, sizeY, sizeZ;
	int ptype;
	int pblock;
	int xPencilsPerSlab, yPencilsPerSlab, zPencilsPerSlab;
	int xsquare[2], ysquare[2], zsquare[2];
	complex *dataPtr;
 private:
	void init(int sizex, int sizey, int sizez, int _ptype, int _pblock, complex *dptr,  int _xPencilsPerSlab, int _yPencilsPerSlab, int _zPencilsPerSlab) {
	    if (sizex != sizey || sizey != sizez)
			ckerr << "WARNING"
				  << "This configuration of the source and destination "
				  << "is not consistent, check the dimensions. The program is "
				  << "likely to misbehave" 
				  << endl;
	    ptype = _ptype;
	    pblock = _pblock;
	    xPencilsPerSlab = _xPencilsPerSlab; 
	    yPencilsPerSlab = _yPencilsPerSlab;
	    zPencilsPerSlab = _zPencilsPerSlab;
	    dataPtr = dptr;
	    sizeX = sizex;
	    sizeY = sizey;
	    sizeZ = sizez;
	    
	    CkAssert((sizeX==sizeY) && (sizeX==sizeZ));
	    
	    if(pblock == PencilBlock::SQUAREBLOCK){
		getSquaresize(xPencilsPerSlab, sizeX, xsquare);
		getSquaresize(yPencilsPerSlab, sizeX, ysquare);
		getSquaresize(zPencilsPerSlab, sizeX, zsquare);
		
		CkAssert((xsquare[1]%ysquare[1]==0) || (ysquare[1]%xsquare[1]==0));
		CkAssert((zsquare[0]%ysquare[0]==0) || (ysquare[0]%zsquare[0]==0));
	    }
	    else {
		xsquare[0]=xPencilsPerSlab;  xsquare[1]=1;
		ysquare[0]=yPencilsPerSlab;  ysquare[1]=1;
		zsquare[0]=zPencilsPerSlab;  zsquare[1]=1;
		CkAssert((yPencilsPerSlab%zPencilsPerSlab==0) 
			 || (zPencilsPerSlab%yPencilsPerSlab==0));
	    }
	}
	void getSquaresize(int size, int planesize, int *square) {
	    int squaresize = (int)sqrt((float)size);
	    if(size==squaresize*squaresize){
		square[0]=squaresize;
		square[1]=squaresize;
	    }
	    else{
		while(squaresize>1 && ((size%squaresize!=0) || 
		       ((size%squaresize==0)&&(planesize%squaresize!=0||planesize%(size/squaresize)!=0)))){
		    squaresize--;
		}
		square[1]=squaresize;
		square[0]=size/squaresize;
	    }
	    CkAssert(size==square[0]*square[1]);
	}
};

#include "fftlib.decl.h"

PUPmarshall(NormalFFTinfo)
PUPmarshall(LineFFTinfo)


typedef struct _SlabArrayInfo{
    int count;
    NormalFFTinfo info;
//    SlabArrayInfo() {count=0; info=NormalFFTinfo();}
//    ~SlabArrayInfo() {}
}SlabArrayInfo;

/*
 * Abstract super class for the two dimensional array of slabs,
 * used for doing 3D FFT
 */
class SlabArray: public CBase_SlabArray {
 public:
	SlabArray(CkMigrateMessage *m) {}
	SlabArray() {}
	virtual ~SlabArray() {}

	// user SHOULD redefine these
	virtual void doneFFT(int id) {
	    ckout << "NormalSlabArray finished FFT" << endl; 
	    CkExit();
	}
	virtual void doneIFFT(int id) {
	    ckout << "NormalSlabArray finished IFFT" << endl;
	    CkExit();
	}

	// Library subclasses MUST define these
	virtual void doFFT(int, int) = 0; //ffts data from src to dest
	virtual void doIFFT(int, int) = 0; //ffts data from dest to src
 protected:
	CProxy_SlabArray srcProxy, destProxy;
	CkVec<SlabArrayInfo*> infoVec;
};

/*
 * Normal, since the data is expected to be present all through the slab
 */

class NormalSlabArray: public CBase_NormalSlabArray {
 public:
	NormalSlabArray(CkMigrateMessage *m): CBase_NormalSlabArray(m) {CkPrintf("migrate constructor called\n");}
	NormalSlabArray() {
#if VERBOSE
	    CkPrintf("Empty constructor called\n");
#endif
	    fwd2DPlan = bwd2DPlan = (fftwnd_plan) NULL;
	    fwd1DPlan = bwd1DPlan = (fftw_plan) NULL;
	}

	NormalSlabArray(NormalFFTinfo &info, 
			CProxy_NormalSlabArray src, CProxy_NormalSlabArray dest);
	~NormalSlabArray();


	void acceptDataForFFT(int, complex *, int, int);
	void acceptDataForIFFT(int, complex *, int, int);

	void doFFT(int src_id = 0, int dst_id = 0);
	void doIFFT(int src_id = 0, int dst_id = 0);

	void pup(PUP::er &p);

	void setup(NormalFFTinfo &info, 
		   CProxy_NormalSlabArray src, CProxy_NormalSlabArray dest);
protected:
	fftwnd_plan fwd2DPlan, bwd2DPlan;
	fftw_plan fwd1DPlan, bwd1DPlan;

	void createPlans(NormalFFTinfo &info);
};

class NormalRealSlabArray: public CBase_NormalRealSlabArray {
 public:
	NormalRealSlabArray(CkMigrateMessage *m): CBase_NormalRealSlabArray(m) {}
	NormalRealSlabArray() {
#if VERBOSE
	    CkPrintf("Empty constructor called\n");
#endif
	    tempdataPtr = NULL;
	    rfwd1DXPlan = rbwd1DXPlan = (rfftw_plan) NULL;
	    fwd1DYPlan = bwd1DYPlan = (fftw_plan) NULL;
	    fwd1DZPlan = bwd1DZPlan = (fftw_plan) NULL;
	    rfwd2DXYPlan = rfwd2DXYPlan = (rfftwnd_plan)NULL;
	}

	NormalRealSlabArray(NormalFFTinfo &info,
			    CProxy_NormalRealSlabArray, CProxy_NormalRealSlabArray);
	~NormalRealSlabArray();


	void acceptDataForFFT(int, complex *, int, int);
	void acceptDataForIFFT(int, complex *, int, int);

	void doFFT(int src_id = 0, int dst_id = 0);
	void doIFFT(int src_id = 0, int dst_id = 0);

	void pup(PUP::er &p);

	void createPlans(NormalFFTinfo &info);

 protected:
	rfftwnd_plan rfwd2DXYPlan, rbwd2DXYPlan;
	rfftw_plan rfwd1DXPlan, rbwd1DXPlan;
	fftw_plan fwd1DYPlan, bwd1DYPlan; 
	fftw_plan fwd1DZPlan, bwd1DZPlan;

	NormalFFTinfo *fftinfos[MAX_FFTS];
	
	void setup(NormalFFTinfo &info, 
		   CProxy_NormalRealSlabArray, CProxy_NormalRealSlabArray);

 private:
	complex *tempdataPtr;
};

#if 0
/* 
 * This represents a run of data in space in a particular direction.
 * The indices may be increasing/decreasing in that direction.
 *
 * x, y, z <int> - starting position of the run
 * length <int> - 
 * dir <DIRECTION> - direction of the run in space
 * inc <(+1,-1)> - // +1 if incremental run, -1 if decremental
 */
class RunDescriptor {
 public:
	static const short X = 1;
	static const short Y = 2;	
	static const short Z = 2;

	//constructors
	RunDescriptor(int x_, int y_, int z_, int l_, int i_ = 0) {x = x_; y = y_; z = z_; length = l_; inc = i_;}
	RunDescriptor(const RunDescriptor &rd) {x = rd.x; y = rd.y; z = rd.z; length = rd.length; inc = rd.inc;}

	// charm pup routine
	void pup(PUP::er &p) {p|x; p|y; p|z; p|length; p(dir); p|inc;}

	// getting the offset of the (i+1)-th point of the run from its start
	// this takes care of the incremental/decremental nature of the run
	int offset(int i) const { return (inc > 0) ? i : -i; }

	// data
	int x, y, z;
	int length;
	short dir;
	int inc; 
};

/*
 * Sparse, the data is represented as ``runs''
 */
class SparseSlabArray: public CBase_SparseSlabArray {
 public:
	SparseSlabArray(CkMigrateMessage *m): SlabArray(m) {}
	SparseSlabArray(): SlabArray() {}
	SparseSlabArray(FFTinfo &info); 
	~SparseSlabArray();
  
	virtual void getRuns(complex **runs, int *numRuns, int *numPoints) const {*runs = NULL; *numRuns = 0; *numPoints = 0;}

 private:
	int count;
	fftwnd_plan fwd2DPlan, bwd2DPlan;
	fftw_plan fwd1DPlan, bwd1DPlan;
	FFTinfo info;
	int *nonZero;
};
#endif

typedef struct _PencilArrayInfo{
    int count;
    LineFFTinfo info;
}PencilArrayInfo;

class SendFFTMsg: public CMessage_SendFFTMsg {
public:
    int size;
    int id;
    int direction;
    int ypos;
    int zpos;
    complex *data;
};


class NormalLineArray : public CBase_NormalLineArray {
 public:
    NormalLineArray (CkMigrateMessage *m) {}
    NormalLineArray () {
	fwdplan = bwdplan =NULL;
	id = -1;
	line = NULL;
    }
    NormalLineArray (LineFFTinfo &info, CProxy_NormalLineArray _xProxy, CProxy_NormalLineArray _yProxy, CProxy_NormalLineArray _zProxy);
    ~NormalLineArray () {}
    void setup (LineFFTinfo &info, CProxy_NormalLineArray _xProxy, CProxy_NormalLineArray _yProxy, CProxy_NormalLineArray _zProxy);
    void doFirstFFT(int id, int direction);
    void doSecondFFT(int ypos, complex *val, int size, int id, int direction);
    void doThirdFFT(int zpos, int ypos, complex *val, int size, int id, int direction);

    void doSecondFFT(SendFFTMsg *msg);
    void doThirdFFT(SendFFTMsg *msg);

    void doFFT(int id, int direction) {doFirstFFT(id, direction);}
    virtual void doneFFT(int id, int direction);
    void setInstance(int id_) { id = id_; 
    contribute(sizeof(int), &id_, CkReduction::sum_int);
    }
 protected:
    complex *line;
    fftw_plan fwdplan, bwdplan;
    int id;

    CProxy_NormalLineArray xProxy, yProxy, zProxy;
    CkVec<PencilArrayInfo*> infoVec;
};


#define CAREFUL 1

#endif //_fftlib_h_
