#include "fftlib.h"

/* 
 * This is on the "source" side.
 * there are srcSize[0] rows of size srcSize[1] each, in row-major order
 */
void
NormalRealSlabArray::doFFT(int src_id, int dst_id)
{
    NormalFFTinfo &fftinfo = (infoVec[src_id]->info);
    
    if(fftinfo.transformType != REAL_TO_COMPLEX) CkPrintf("Transform Type at doFFT is %d\n", fftinfo.transformType);
    CkAssert(fftinfo.transformType == REAL_TO_COMPLEX);
    
    int rplaneSize = fftinfo.srcSize[0] * fftinfo.srcSize[1];
    int cplaneSize = (fftinfo.srcSize[0]/2+1) * fftinfo.srcSize[1];
    int lineSize = fftinfo.srcSize[1];

    double *dataPtr = (double*)fftinfo.dataPtr;
    complex *outData = new complex[cplaneSize * fftinfo.srcPlanesPerSlab];
    complex *outDataPtr = outData;
    complex *outData2 = new complex[cplaneSize * fftinfo.srcPlanesPerSlab];
    complex *outDataPtr2 = outData2;

	// do the 2D forward ffts
    CmiAssert(rfwd1DXPlan!=NULL && fwd1DYPlan!=NULL);
    int p;

    for(p = 0; p < fftinfo.srcPlanesPerSlab; p++){
        // real_to_complex on Y dimension
	rfftwnd_one_real_to_complex(rfwd2DXYPlan, (fftw_real*)dataPtr,
				    (fftw_complex*)outDataPtr2);

/*

	if(thisIndex==0 && src_id==0){
	    for(int i=0;i<fftinfo.srcSize[1]*fftinfo.srcSize[0]; i++)
		CkPrintf("%d %g\n", i, dataPtr[i]);
	}

	fftw(fwd1DYPlan, fftinfo.srcSize[0]/2+1,
	     (fftw_complex*)outDataPtr, fftinfo.srcSize[1]/2+1, 1, 
	     (fftw_complex*)outDataPtr2, fftinfo.srcSize[1]/2+1, 1);


	if(thisIndex==0 && src_id==0){
	    for(int i=0;i<(fftinfo.srcSize[1]/2+1)*fftinfo.srcSize[0]; i++)
		CkPrintf("%d %g %g\n", i, outDataPtr2[i].re, outDataPtr2[i].im);
	}
*/
	dataPtr += rplaneSize;
	outDataPtr += cplaneSize;
	outDataPtr2 += cplaneSize;
    }
    
    dataPtr = (double*)fftinfo.dataPtr;
    outDataPtr2 = outData2;


    // allocating the data for sending to destination side
    lineSize = fftinfo.srcSize[1]/2+1;
    complex *sendData = new complex[fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab * lineSize];
    complex *temp;
    // i <> pe if destPlanesPerSlab is not 1
    int pe, i, j;
    for(i = 0, pe = 0; i < fftinfo.srcSize[0]; i += fftinfo.destPlanesPerSlab, pe++) {
        int ti;
        temp = sendData;
// Copy the data in (y-z-x) order
        for(ti = i; ti < i + fftinfo.destPlanesPerSlab; ti++) // x direction
            for(p = 0; p < fftinfo.srcPlanesPerSlab; p++) {   // z direction
                memcpy(temp,
                       outDataPtr2 + p * cplaneSize + ti * lineSize,
                       sizeof(complex) * lineSize);
                temp += lineSize;
            }
#ifdef VERBOSE
        CkPrintf("[%d] sendFFTData to %d, size %d \n", thisIndex, pe, 
		 lineSize * fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab);
#endif
        ((CProxy_NormalSlabArray)destProxy)(pe).acceptDataForFFT(lineSize * fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab, sendData, thisIndex, dst_id);
    }
    delete [] sendData;
    delete [] outData;
    delete [] outData2;
}

/*
 * This is on the "destination" side.
 */
void
NormalRealSlabArray::acceptDataForFFT(int numPoints, complex *points, int posn, int info_id)
{
    NormalFFTinfo &fftinfo = (infoVec[info_id]->info);
    CkAssert(fftinfo.transformType == COMPLEX_TO_REAL);

    complex *dataPtr = (complex*)fftinfo.dataPtr;
    int lineSize = fftinfo.destSize[1]/2+1;
    
#if CAREFUL
    CkAssert(numPoints == fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab * lineSize);
#endif
	
    infoVec[info_id]->count++;
    int planeSize = fftinfo.destSize[0] * (fftinfo.destSize[1]/2+1);
    int p;
    for (p = 0; p < fftinfo.destPlanesPerSlab; p++) {
	memcpy(dataPtr + posn * fftinfo.srcPlanesPerSlab * lineSize + p * planeSize,
	       points, 
	       sizeof(complex) * lineSize * fftinfo.srcPlanesPerSlab);
	points += lineSize * fftinfo.srcPlanesPerSlab;
    }

    if (infoVec[info_id]->count == fftinfo.destSize[0] / fftinfo.srcPlanesPerSlab) {
	infoVec[info_id]->count = 0;
	CkAssert(fwd1DZPlan != NULL);
	for(p = 0; p < fftinfo.destPlanesPerSlab; p++) {
		fftw(fwd1DZPlan, 
		     lineSize,
		     (fftw_complex*)dataPtr + p * planeSize,
		     lineSize, 1, //stride, nextFFT
		     NULL, 0, 0);
	}
	doneFFT(info_id);
    }
}

/*
 * This is on the "destination" side.
 *    There are destSize[0] rows of destSize[1] each, in row major order, and srcSize[1] == destSize[1]
 */
void
NormalRealSlabArray::doIFFT(int src_id, int dst_id)
{
    NormalFFTinfo &fftinfo = (infoVec[src_id]->info);
    CkAssert(fftinfo.transformType == COMPLEX_TO_REAL);

    complex *dataPtr = (complex*)fftinfo.dataPtr;
    int planeSize = fftinfo.destSize[0] * (fftinfo.destSize[1]/2+1);
    int lineSize = fftinfo.destSize[1]/2+1;
    
    CmiAssert(bwd1DZPlan!=NULL && bwd1DYPlan!=NULL);
    int p;

    for(p = 0; p < fftinfo.destPlanesPerSlab; p++){
	// Z direction FFT
	fftw(bwd1DZPlan, lineSize, 
	     (fftw_complex*)dataPtr+p*planeSize, lineSize, 1, 
	     NULL, 0, 0);

/*	if(thisIndex==0 && src_id==0){
	    for(int i=0;i<(fftinfo.srcSize[1]/2+1)*fftinfo.srcSize[0]; i++)
		CkPrintf("%d %g %g\n", i, dataPtr[i+p*planeSize].re, dataPtr[i+p*planeSize].im);
	}
*/
    }
      
    complex *sendData = new complex[fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab * lineSize];
    complex *temp;
    int i, pe;
    for (i = 0, pe = 0; i < fftinfo.destSize[0]; i += fftinfo.srcPlanesPerSlab, pe++) {
	int ti;
	temp = sendData;
	// Sending in (y-x-z) order
	for (ti = i; ti < i + fftinfo.srcPlanesPerSlab; ti++)
	    for (p = 0; p < fftinfo.destPlanesPerSlab; p++) {
		memcpy(temp,
		       dataPtr + p * planeSize + ti * lineSize,
		       sizeof(complex) * lineSize);
		temp += lineSize;
	    }

	((CProxy_NormalSlabArray)srcProxy)(pe).acceptDataForIFFT(lineSize * fftinfo.destPlanesPerSlab * fftinfo.srcPlanesPerSlab, sendData, thisIndex, dst_id);
    }
    delete [] sendData;
}

void
NormalRealSlabArray::acceptDataForIFFT(int numPoints, complex *points, int posn, int info_id)
{
    NormalFFTinfo &fftinfo = (infoVec[info_id]->info);
    CkAssert(fftinfo.transformType == REAL_TO_COMPLEX);

    int rplaneSize = fftinfo.srcSize[0] * fftinfo.srcSize[1];
    int cplaneSize = fftinfo.srcSize[0] * (fftinfo.srcSize[1]/2+1);
    int planeSize = fftinfo.destSize[0] * (fftinfo.destSize[1]/2+1);
    int lineSize = fftinfo.destSize[1]/2+1;
    if(tempdataPtr==NULL)
	tempdataPtr = new complex[fftinfo.srcPlanesPerSlab * cplaneSize];
    complex *inDataPtr = tempdataPtr;
#if CAREFUL
    CkAssert(numPoints == fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab * lineSize);
#endif

    infoVec[info_id]->count++;
    int p;
    complex *pointPtr = points;
    inDataPtr = tempdataPtr + posn * lineSize * fftinfo.destPlanesPerSlab;
    for(p = 0; p < fftinfo.srcPlanesPerSlab; p++) {
	memcpy(inDataPtr, pointPtr, lineSize* fftinfo.destPlanesPerSlab*sizeof(complex));
	inDataPtr += planeSize;
	pointPtr += lineSize*fftinfo.destPlanesPerSlab;
    }
    
    int expectedCount = fftinfo.srcSize[0]/fftinfo.destPlanesPerSlab;
    if (infoVec[info_id]->count == expectedCount) {
	infoVec[info_id]->count = 0;
	CmiAssert(rbwd1DXPlan!=NULL);
	double *dataPtr = (double*)fftinfo.dataPtr;
	for(p = 0; p < fftinfo.srcPlanesPerSlab; p++) {
	    rfftwnd_one_complex_to_real(rbwd2DXYPlan, 
					(fftw_complex*)(tempdataPtr+p*cplaneSize), 
					(fftw_real*)(dataPtr+p*rplaneSize));
/*
	    if(thisIndex==0 && info_id==0){
		for(int i=0;i<fftinfo.srcSize[1]*fftinfo.srcSize[0]; i++)
		    CkPrintf("%d %g\n", i, dataPtr[i]);
	    }
	    

	    // Doing X direction first
		fftw(bwd1DYPlan, lineSize, 
		     (fftw_complex*)tempdataPtr+p*planeSize, lineSize, 1,
		     NULL, 0, 0);
		
	    // complex_to_real transform at Y direction
		rfftw(rbwd1DXPlan,
		      fftinfo.srcSize[0],
		      (fftw_real*)(tempdataPtr + p * cplaneSize),
		      1, lineSize*2, //stride, nextFFT
		      (fftw_real*)dataPtr + p * rplaneSize, 
		      1, fftinfo.srcSize[1]);
*/
	}
	doneIFFT(info_id);
    }
}
void NormalRealSlabArray::createPlans(NormalFFTinfo &info)
{
    if (info.isSrcSlab) {
	int size[]={info.srcSize[0], info.srcSize[1]};
	rfwd2DXYPlan = rfftw2d_create_plan(info.srcSize[0], info.srcSize[1], FFTW_REAL_TO_COMPLEX, FFTW_OUT_OF_PLACE);
	rbwd2DXYPlan = rfftw2d_create_plan(info.srcSize[0], info.srcSize[1], FFTW_COMPLEX_TO_REAL, FFTW_OUT_OF_PLACE);

	rfwd1DXPlan = rfftw_create_plan(info.srcSize[0], FFTW_REAL_TO_COMPLEX, FFTW_OUT_OF_PLACE);
	//fwd1DYPlan = fftw_create_plan(info.srcSize[1], FFTW_BACKWARD, FFTW_MEASURE|FFTW_IN_PLACE); 
	fwd1DYPlan = fftw_create_plan(info.srcSize[1], FFTW_BACKWARD, FFTW_OUT_OF_PLACE); 
	rbwd1DXPlan = rfftw_create_plan(info.srcSize[0], FFTW_COMPLEX_TO_REAL, FFTW_OUT_OF_PLACE);
	bwd1DYPlan = fftw_create_plan(info.destSize[1], FFTW_BACKWARD, FFTW_IN_PLACE);
    }
    else {

	bwd1DZPlan = fftw_create_plan(info.destSize[0], FFTW_BACKWARD, FFTW_IN_PLACE);
	bwd1DYPlan = fftw_create_plan(info.destSize[1], FFTW_BACKWARD, FFTW_IN_PLACE);
	fwd1DZPlan = fftw_create_plan(info.destSize[0], FFTW_FORWARD, FFTW_IN_PLACE);
    }
}

void NormalRealSlabArray::setup(NormalFFTinfo &info,
				CProxy_NormalRealSlabArray src, 
				CProxy_NormalRealSlabArray dest)
{
    SlabArrayInfo *slabinfo=new SlabArrayInfo;
    slabinfo->info = info;
    slabinfo->count = 0;
    infoVec.insert(infoVec.size(), slabinfo);

    srcProxy = src;
    destProxy = dest;
    rfwd1DXPlan = rbwd1DXPlan = (rfftw_plan) NULL;
    fwd1DYPlan = bwd1DYPlan = (fftw_plan) NULL;
    fwd1DZPlan = bwd1DZPlan = (fftw_plan) NULL;

    createPlans(info);
}

NormalRealSlabArray::NormalRealSlabArray(NormalFFTinfo &info,
					 CProxy_NormalRealSlabArray src, 
					 CProxy_NormalRealSlabArray dest)
{ 
    setup(info, src, dest);
}

NormalRealSlabArray::~NormalRealSlabArray() 
{
    if (rfwd1DXPlan)
	rfftw_destroy_plan(rfwd1DXPlan);
    if (rbwd1DXPlan)
	rfftw_destroy_plan(rbwd1DXPlan);
    if (fwd1DYPlan)
	fftw_destroy_plan(fwd1DYPlan);
    if (bwd1DYPlan)
	fftw_destroy_plan(bwd1DYPlan);
    if (fwd1DZPlan)
	fftw_destroy_plan(fwd1DZPlan);
    if (bwd1DZPlan)
	fftw_destroy_plan(bwd1DZPlan);

    if(rfwd2DXYPlan)
	fftwnd_destroy_plan(rfwd2DXYPlan);
    if(rbwd2DXYPlan)
	fftwnd_destroy_plan(rbwd2DXYPlan);

    infoVec.removeAll();
    if(tempdataPtr != NULL) 
	delete [] tempdataPtr;
}


void NormalRealSlabArray::pup(PUP::er &p)
{
    int i;

/*    for (i = 0; i < MAX_FFTS; i++) {
	if (p.isUnpacking()) {
          fftinfos[i] = NULL;
	}
	int val = fftinfos[i]? 1:0;
	p | val; 
	if (val) {                // if this entry is not NULL
	    if (p.isUnpacking()){
		fftinfos[i] = new NormalFFTinfo;
	    }  
	    fftinfos[i]->pup(p);
	    
            if (p.isUnpacking()){
		createPlans(*fftinfos[i]);
	    }	    
	}
    }
*/  
}


