#ifndef _fftlib_h_
#define _fftlib_h_

#include <charm++.h>
#include "ckcomplex.h"


#include "fftlib.decl.h"

#define MAX_FFTS 5

class NormalFFTinfo {
 public:
	// Constructors
	NormalFFTinfo(CProxy_NormalSlabArray &sProxy, CProxy_NormalSlabArray &dProxy, int sDim[2], int dDim[2], int isSrc, complex *dptr, int sPlanesPerSlab=1, int dPlanesPerSlab=1) {
		init(sProxy, dProxy, sDim, dDim, isSrc, dptr, sPlanesPerSlab, dPlanesPerSlab);
	}

	NormalFFTinfo(NormalFFTinfo &info) {
		init(info.srcProxy, info.destProxy, info.srcSize, info.destSize, info.isSrcSlab, (complex *) NULL, info.srcPlanesPerSlab, info.destPlanesPerSlab);
	}
	NormalFFTinfo(void) {}

	// charm pup function
	void pup(PUP::er &p) {
		p|srcProxy;
		p|destProxy;
		p(srcSize, 2);
		p(destSize, 2);
		p(isSrcSlab);
		p(srcPlanesPerSlab);
		p(destPlanesPerSlab);
		if (p.isUnpacking()) 
			dataPtr = (complex *) NULL;
	}

	CProxy_NormalSlabArray srcProxy, destProxy;
	int srcSize[2], destSize[2];
	bool isSrcSlab;
	int srcPlanesPerSlab, destPlanesPerSlab;
	complex *dataPtr;
 private:
	void init(CProxy_NormalSlabArray &sProxy, CProxy_NormalSlabArray &dProxy, int sDim[2], int dDim[2], int isSrc, complex *dptr, int sPlanesPerSlab, int dPlanesPerSlab) {
		if (sDim[1] != dDim[1])
			ckerr << "WARNING"
				  << "This configuration of the source and destination "
				  << "is not consistent, check the dimensions. The program is "
				  << "likely to misbehave" 
				  << endl;
		srcProxy = sProxy; 
		destProxy = dProxy;
		isSrcSlab = isSrc;
		srcPlanesPerSlab = sPlanesPerSlab; 
		destPlanesPerSlab = dPlanesPerSlab;
		dataPtr = dptr;
		memcpy(srcSize, sDim, 2 * sizeof(int));
		memcpy(destSize, dDim, 2 * sizeof(int));
	}
};

class LineFFTinfo {
 public:
	// Constructors
	LineFFTinfo(CProxy_NormalLineArray &xProxy, CProxy_NormalLineArray &yProxy, CProxy_NormalLineArray &zProxy, int size[3], int isSrc, complex *dptr, int sPencilsPerSlab=1, int dPencilsPerSlab=1) {
		init(xProxy, yProxy, zProxy, size[0], size[1], size[2], isSrc, dptr, sPencilsPerSlab, dPencilsPerSlab);
	}
	LineFFTinfo(LineFFTinfo &info) {
		init(info.xlinesProxy, info.ylinesProxy, info.zlinesProxy, info.sizeX, info.sizeY, info.sizeZ, info.isSrcSlab, (complex *) NULL, info.sPencilsPerSlab, info.dPencilsPerSlab);
	}
	LineFFTinfo(void) {}

	// charm pup function
	void pup(PUP::er &p) {
		p|xlinesProxy;
		p|ylinesProxy;
		p|zlinesProxy;
		p|sizeX;
		p|sizeY;
		p|sizeZ;
		p(isSrcSlab);
		p(sPencilsPerSlab);
		p(dPencilsPerSlab);
		if (p.isUnpacking()) 
			dataPtr = (complex *) NULL;
	}
	CProxy_NormalLineArray xlinesProxy, ylinesProxy, zlinesProxy;
	int sizeX, sizeY, sizeZ;
	bool isSrcSlab;
	int sPencilsPerSlab, dPencilsPerSlab;
	complex *dataPtr;
 private:
	void init(CProxy_NormalLineArray &xProxy, CProxy_NormalLineArray &yProxy, CProxy_NormalLineArray &zProxy, int sizex, int sizey, int sizez, int isSrc, complex *dptr, int _sPencilsPerSlab, int _dPencilsPerSlab) {
		if (sizex != sizey || sizey != sizez)
			ckerr << "WARNING"
				  << "This configuration of the source and destination "
				  << "is not consistent, check the dimensions. The program is "
				  << "likely to misbehave" 
				  << endl;
		xlinesProxy = xProxy; 
		ylinesProxy = yProxy;
		zlinesProxy = zProxy;
		isSrcSlab = isSrc;
		sPencilsPerSlab = _sPencilsPerSlab; 
		dPencilsPerSlab = _dPencilsPerSlab;
		dataPtr = dptr;
		sizeX = sizex;
		sizeY = sizey;
		sizeZ = sizez;
	}
};

PUPmarshall(NormalFFTinfo);
PUPmarshall(LineFFTinfo);

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
};

/*
 * Normal, since the data is expected to be present all through the slab
 */

class NormalSlabArray: public SlabArray {
 public:
	NormalSlabArray(CkMigrateMessage *m): SlabArray(m) {}
	NormalSlabArray() {
		int i;
		for (i = 0; i < MAX_FFTS; i++) {
			fftinfos[i] = NULL;
			counts[i] = 0;
		}
		fwd2DPlan = bwd2DPlan = (fftwnd_plan) NULL;
		fwd1DPlan = bwd1DPlan = (fftw_plan) NULL;
	}

	NormalSlabArray(NormalFFTinfo &info); 
	~NormalSlabArray();

	void acceptDataForFFT(int, complex *, int, int);
	void acceptDataForIFFT(int, complex *, int, int);

	void doFFT(int src_id = 0, int dst_id = 0);
	void doIFFT(int src_id = 0, int dst_id = 0);

	void pup(PUP::er &p);

 protected:
	fftwnd_plan fwd2DPlan, bwd2DPlan;
	fftw_plan fwd1DPlan, bwd1DPlan;
	NormalFFTinfo *fftinfos[MAX_FFTS];
 private:
	int counts[MAX_FFTS];
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
class SparseSlabArray: public SlabArray {
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

class NormalLineArray : public CBase_NormalLineArray {
 public:
    NormalLineArray (CkMigrateMessage *m) {}
    NormalLineArray () {
	id = -1;
	for(int i=0; i<MAX_FFTS; i++)
	    count[i] = 0;
	line = NULL;
    }
    NormalLineArray (LineFFTinfo &info, int _flag) {
	int sizeX = info.sizeX;
	int sizeY = info.sizeY;
	fftinfos[0] = new LineFFTinfo(info);
	line = NULL;
	fwdplan = fftw_create_plan(sizeX, FFTW_FORWARD, FFTW_USE_WISDOM|FFTW_IN_PLACE|FFTW_MEASURE);
	bwdplan = fftw_create_plan(sizeY, FFTW_BACKWARD, FFTW_USE_WISDOM|FFTW_IN_PLACE|FFTW_MEASURE);
	id = -1;
	for(int i=0; i<MAX_FFTS; i++)
	    count[i] = 0;
	flag = _flag;
    }
    ~NormalLineArray () {}
    void doFirstFFT(int id, int direction);
    void doSecondFFT(int ypos, complex *val, int size, int id, int direction);
    void doThirdFFT(int zpos, int ypos, complex *val, int size, int id, int direction);
    virtual void doneFFT(int id, int direction);
    void setInstance(int id_) { id = id_; 
    //mgrProxy.ckLocalBranch()->registerElement(id); 
    contribute(sizeof(int), &id_, CkReduction::sum_int);
    }
 protected:
    int flag;
    complex *line;
    fftw_plan fwdplan, bwdplan;
    int id;
    LineFFTinfo *fftinfos[MAX_FFTS];
    int count[MAX_FFTS];
};


#define CAREFUL 1

#endif //_fftlib_h_
