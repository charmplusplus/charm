#include <charm++.h>
#include "util.h"

#ifndef _fftlib_h_
#define _fftlib_h_

#include "fftlib.decl.h"

template <class T>
class FFTinfo {
 public:
	// Constructors
	FFTinfo(T &sProxy, T &dProxy, int sDim[2], int dDim[2], int isSrc, complex *dptr, int sPlanesPerSlab=1, int dPlanesPerSlab=1) {
		init(sProxy, dProxy, sDim, dDim, isSrc, dptr, sPlanesPerSlab, dPlanesPerSlab);
	}
	FFTinfo(FFTinfo<T> &info) {
		init(info.srcProxy, info.destProxy, info.srcSize, info.destSize, info.isSrcSlab, (complex *) NULL, info.srcPlanesPerSlab, info.destPlanesPerSlab);
	}
	FFTinfo(void) {}

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

	T srcProxy, destProxy;
	int srcSize[2], destSize[2];
	bool isSrcSlab;
	int srcPlanesPerSlab, destPlanesPerSlab;
	complex *dataPtr;
 private:
	void init(T &sProxy, T &dProxy, int sDim[2], int dDim[2], int isSrc, complex *dptr, int sPlanesPerSlab, int dPlanesPerSlab) {
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

typedef FFTinfo<CProxy_NormalSlabArray> NormalFFTinfo;

PUPmarshall(NormalFFTinfo);

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
	 static const short MAX_FFTS = 4;
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
#if 0
	NormalSlabArray(NormalFFTinfo &info); 
#endif
	~NormalSlabArray();

	void acceptDataForFFT(int, complex *, int, int);
	void acceptDataForIFFT(int, complex *, int, int);

	void doFFT(int src_id = 0, int dst_id = 0);
	void doIFFT(int src_id = 0, int dst_id = 0);
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
#define CAREFUL 1

#endif //_fftlib_h_
