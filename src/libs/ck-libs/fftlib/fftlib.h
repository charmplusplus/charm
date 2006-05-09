#ifndef _fftlib_h_
#define _fftlib_h_

#include <charm++.h>
#include "ckcomplex.h"
#include "EachToManyMulticastStrategy.h"
#include "StreamingStrategy.h"
#include "comlib.h"

// Define which FFT sequential library to use
// Note : One flag and one flag only is set to 1
#define FFT_USE_FFTW_2_1_5 1 
#define FFT_USE_FFTW_3_1   0
#define FFT_USE_ESSL       0 
#define FFT_USE_NETLIB     0
#define FFT_USE_FFTE	   0 

#if FFT_USE_FFTW_2_1_5 
#include "drfftw.h"
#elif  FFT_USE_ESSL
#include "essl.h"
#endif

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
	LineFFTinfo(int size[3], int _ptype, int _pblock,ckcomplex  *dptr, int _xPencilsPerSlab=1, int _yPencilsPerSlab=1, int _zPencilsPerSlab=1) {
	  init(size[0], size[1], size[2], _ptype, _pblock, dptr, _xPencilsPerSlab, _yPencilsPerSlab, _zPencilsPerSlab);
	}
	LineFFTinfo(LineFFTinfo &info) {
	    init(info.sizeX, info.sizeY, info.sizeZ, info.ptype, info.pblock, (ckcomplex *) NULL, info.xPencilsPerSlab, info.yPencilsPerSlab, info.zPencilsPerSlab);
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
		dataPtr = (ckcomplex *) NULL;
	}
	int sizeX, sizeY, sizeZ;
	int ptype;
	int pblock;
	int xPencilsPerSlab, yPencilsPerSlab, zPencilsPerSlab;
	int xsquare[2], ysquare[2], zsquare[2];
	ckcomplex *dataPtr;
 private:
	void init(int sizex, int sizey, int sizez, int _ptype, int _pblock,ckcomplex  *dptr,  int _xPencilsPerSlab, int _yPencilsPerSlab, int _zPencilsPerSlab) {
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

PUPmarshall(NormalFFTinfo);
PUPmarshall(LineFFTinfo);


typedef struct _SlabArrayInfo{
    int count;
    NormalFFTinfo info;
    ComlibInstanceHandle fftcommInstance;
//    SlabArrayInfo() {count=0; info=NormalFFTinfo();}
//    ~SlabArrayInfo() {}
}SlabArrayInfo;

#define ESSL_CFT dcft
#define ESSL_CFT2 dcft2

#if FFT_USE_FFTW_2_1_5 
#define FFT1_ONE(plan,x)  fftw_one(plan,(fftw_complex*)x,NULL);  
#elif FFT_USE_FFTW_3_1 
#define FFT1_ONE(plan,x)  
#elif FFT_USE_ESSL 
#define FFT1_ONE(plan,x)  ESSL_CFT(plan->init,x,plan->incx1,plan->incx2,x,plan->incx1,plan->incx2,plan->n,1,plan->isign,plan->scale,plan->aux1,plan->naux1,plan->aux2,plan->naux2);
#endif 

#if FFT_USE_FFTW_2_1_5 
#define FFT1_MANY(plan,N,x) fftw(plan,N,(fftw_complex*)x,N,1,NULL,1,0);  
#elif FFT_USE_FFTW_3_1 
#define FFT1_MANY(plan,N,x)  
#elif FFT_USE_ESSL 
#define FFT1_MANY(plan,N,x)  ESSL_CFT(plan->init,x,plan->incx1,plan->incx2,x,plan->incx1,plan->incx2,plan->n, plan->m,plan->isign,plan->scale,plan->aux1,plan->naux1,plan->aux2,plan->naux2);
#endif 

#if FFT_USE_FFTW_2_1_5  
#define FFT2_ONE(plan,x)  fftwnd_one(plan,(fftw_complex*)x,NULL);  
#elif FFT_USE_FFTW_3_1 
#define FFT2_ONE(plan,x)
#elif FFT_USE_ESSL 
#define FFT2_ONE(plan,x)  ESSL_CFT2(plan->init,x,plan->incx1,plan->incx2,x,plan->incx1,plan->incx2,plan->n,plan->m,plan->isign,plan->scale,plan->aux1,plan->naux1,plan->aux2,plan->naux2);
#endif

#if FFT_USE_ESSL
struct essl_fft_plan_struct {
	int init;
	int incx1, incx2;
	int n,m;
	int isign;
	double scale;
	int naux1, naux2;
	double *aux1, *aux2;
};

typedef struct essl_fft_plan_struct *essl_fft_plan;

essl_fft_plan essl_create_1Dplan_one(int _incx1, int _incx2, int _n, int _isign){
	essl_fft_plan plan=new essl_fft_plan_struct();
	plan->incx1=_incx1; plan->incx2=_incx2;
	plan->n=_n; plan->m=1;
	plan->isign=_isign;
	//if(plan->isign>0) 
		plan->scale = 1;
	//else if(plan->isign<0) 
	//	plan->scale = 1.0/(double)(plan->n);
	//else 
	//	CkAbort("ESSL ISIGN CANNOT BE 0!!!\n");
	plan->naux1 = 512;
	plan->aux1 = (double*) (new ckcomplex[plan->naux1]);
	plan->naux2 = 512;
	plan->aux2 = (double*) (new ckcomplex[plan->naux2]);

	// Initialize the aux1 storage
	plan->init=1;
	FFT1_ONE(plan,NULL);
	plan->init=0;
	return plan;
}
essl_fft_plan essl_create_1Dplan_many(int _incx1, int _incx2, int _n, int _m, int _isign){
        essl_fft_plan plan=new essl_fft_plan_struct();
        plan->incx1=_incx1; plan->incx2=_incx2;
        plan->n=_n; plan->m=_m;
        plan->isign=_isign;
        //if(plan->isign>0)
                plan->scale = 1;
        //else if(plan->isign<0)
        //        plan->scale = 1.0/(double)(plan->n);
        //else
        //        CkAbort("ESSL ISIGN CANNOT BE 0!!!\n");
        plan->naux1 = 512;
        plan->aux1 = (double*) (new ckcomplex[plan->naux1]);
        plan->naux2 = 512;
        plan->aux2 = (double*) (new ckcomplex[plan->naux2]);

        // Initialize the aux1 storage
        plan->init=1;
        FFT1_MANY(plan,plan->m,NULL);
        plan->init=0;
        return plan;
}

essl_fft_plan essl_create_2Dplan_one(int _incx1, int _incx2, int _n, int _m, int _isign){ 
	essl_fft_plan plan=new essl_fft_plan_struct();
	plan->incx1=_incx1; plan->incx2=_incx2;
	plan->n=_n; plan->m=_m;
	plan->isign=_isign;
	//if(plan->isign>0) 
		plan->scale = 1;
	//else if(plan->isign<0) 
	//	plan->scale = 1.0/(double)(plan->n*plan->m);
	//else 
	//	CkAbort("ESSL ISIGN CANNOT BE 0!!!\n");
	plan->naux1 = 512;
	plan->aux1 = (double*) (new ckcomplex[plan->naux1]);
	plan->naux2 = 512;
	plan->aux2 = (double*) (new ckcomplex[plan->naux2]);

	// Initialize the aux1 storage
	plan->init=1;
	FFT2_ONE(plan,NULL);
	plan->init=0;
	return plan;
}

void essl_destroy_plan(essl_fft_plan plan){
	if(plan->naux1!=0) delete [] plan->aux1;
	if(plan->naux2!=0) delete [] plan->aux2;
	delete plan;
}

#endif



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
	bool fftuseCommlib;
//	ComlibInstanceHandle fftcommInstance;
	CkVec<SlabArrayInfo*> infoVec;
};

/*
 * Normal, since the data is expected to be present all through the slab
 */

class NormalSlabArray: public SlabArray {
 public:
	NormalSlabArray(CkMigrateMessage *m): SlabArray(m) {CkPrintf("migrate constructor called\n");}
	NormalSlabArray() {
#if VERBOSE
	    CkPrintf("Empty constructor called\n");
#endif
	    fwd2DPlan = bwd2DPlan = NULL;
	    fwd1DPlan = bwd1DPlan = NULL;
	    fftuseCommlib = false;
	    //fftcommInstance = ComlibInstanceHandle();
	}

	NormalSlabArray(NormalFFTinfo &info, 
			CProxy_NormalSlabArray src, CProxy_NormalSlabArray dest, 
			bool useCommlib, ComlibInstanceHandle inst);
	~NormalSlabArray();


	void acceptDataForFFT(int,ckcomplex  *, int, int);
	void acceptDataForIFFT(int,ckcomplex  *, int, int);

	void doFFT(int src_id = 0, int dst_id = 0);
	void doIFFT(int src_id = 0, int dst_id = 0);

	void pup(PUP::er &p);

	void setup(NormalFFTinfo &info, 
		   CProxy_NormalSlabArray src, CProxy_NormalSlabArray dest, 
		   bool useCommlib=false, 
		   ComlibInstanceHandle inst=ComlibInstanceHandle());
protected:
#if FFT_USE_FFTW_2_1_5
	fftwnd_plan fwd2DPlan, bwd2DPlan;
	fftw_plan fwd1DPlan, bwd1DPlan;
#elif FFT_USE_FFTW_3_1
#elif FFT_USE_ESSL
	essl_fft_plan fwd2DPlan, bwd2DPlan;
	essl_fft_plan fwd1DPlan, bwd1DPlan;
#endif
	void createPlans(NormalFFTinfo &info);
};

#if 0
class NormalRealSlabArray: public SlabArray {
 public:
	NormalRealSlabArray(CkMigrateMessage *m): SlabArray(m) {}
	NormalRealSlabArray() {
#if VERBOSE
	    CkPrintf("Empty constructor called\n");
#endif
	    tempdataPtr = NULL;
	    rfwd1DXPlan = rbwd1DXPlan = NULL;
	    fwd1DYPlan = bwd1DYPlan = NULL;
	    fwd1DZPlan = bwd1DZPlan = NULL;
	    rfwd2DXYPlan = rfwd2DXYPlan = NULL;
	    fftuseCommlib = false;
	    //fftcommInstance = ComlibInstanceHandle();	
	}

	NormalRealSlabArray(NormalFFTinfo &info,
			    CProxy_NormalRealSlabArray, CProxy_NormalRealSlabArray, 
			    bool useCommlib, ComlibInstanceHandle inst);
	~NormalRealSlabArray();


	void acceptDataForFFT(int,ckcomplex  *, int, int);
	void acceptDataForIFFT(int,ckcomplex  *, int, int);

	void doFFT(int src_id = 0, int dst_id = 0);
	void doIFFT(int src_id = 0, int dst_id = 0);

	void pup(PUP::er &p);

	void createPlans(NormalFFTinfo &info);

 protected:
#if FFT_USE_FFTW
	rfftwnd_plan rfwd2DXYPlan, rbwd2DXYPlan;
	rfftw_plan rfwd1DXPlan, rbwd1DXPlan;
	fftw_plan fwd1DYPlan, bwd1DYPlan; 
	fftw_plan fwd1DZPlan, bwd1DZPlan;
#elif FFT_USE_ESSL
        essl_fft_plan rfwd2DXYPlan, rbwd2DXYPlan;
        essl_fft_plan rfwd1DXPlan, rbwd1DXPlan;
        essl_fft_plan fwd1DYPlan, bwd1DYPlan;
        essl_fft_plan fwd1DZPlan, bwd1DZPlan;
#endif
	NormalFFTinfo *fftinfos[MAX_FFTS];
      bool fftuseCommlib;
      ComlibInstanceHandle fftcommInstance;
	
      void setup(NormalFFTinfo &info, 
		   CProxy_NormalRealSlabArray, CProxy_NormalRealSlabArray, 
		   bool useCommlib, ComlibInstanceHandle inst);

 private:
	ckcomplex *tempdataPtr;
};
#endif

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
  
	virtual void getRuns(ckcomplex **runs, int *numRuns, int *numPoints) const {*runs = NULL; *numRuns = 0; *numPoints = 0;}

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
    ComlibInstanceHandle fftcommInstance;
}PencilArrayInfo;

class SendFFTMsg: public CMessage_SendFFTMsg {
public:
    int size;
    int id;
    int direction;
    int ypos;
    int zpos;
   ckcomplex  *data;
};

#if 0
class NormalLineArray : public CBase_NormalLineArray {
 public:
    NormalLineArray (CkMigrateMessage *m) {}
    NormalLineArray () {
	fwdplan = bwdplan =NULL;
	fftuseCommlib = false;
	id = -1;
	line = NULL;
    }
    NormalLineArray (LineFFTinfo &info, CProxy_NormalLineArray _xProxy, CProxy_NormalLineArray _yProxy, CProxy_NormalLineArray _zProxy, bool useCommlib, ComlibInstanceHandle &inst);
    ~NormalLineArray () {}
    void setup (LineFFTinfo &info, CProxy_NormalLineArray _xProxy, CProxy_NormalLineArray _yProxy, CProxy_NormalLineArray _zProxy, bool useCommlib, ComlibInstanceHandle &inst);
    void doFirstFFT(int id, int direction);
    void doSecondFFT(int ypos,ckcomplex  *val, int size, int id, int direction);
    void doThirdFFT(int zpos, int ypos,ckcomplex  *val, int size, int id, int direction);

    void doSecondFFT(SendFFTMsg *msg);
    void doThirdFFT(SendFFTMsg *msg);

    void doFFT(int id, int direction) {doFirstFFT(id, direction);}
    virtual void doneFFT(int id, int direction);
    void setInstance(int id_) { id = id_; 
    //mgrProxy.ckLocalBranch()->registerElement(id); 
    contribute(sizeof(int), &id_, CkReduction::sum_int);
    }
 protected:
   ckcomplex  *line;
#if FFT_USE_FFTW
    fftw_plan fwdplan, bwdplan;
#elif FFT_USE_ESSL
    essl_fft_plan fwdplan, bwdplan;
#endif
    int id;

    CProxy_NormalLineArray xProxy, yProxy, zProxy;
    bool fftuseCommlib;
    CkVec<PencilArrayInfo*> infoVec;
};
#endif

#define CAREFUL 1

#endif //_fftlib_h_
