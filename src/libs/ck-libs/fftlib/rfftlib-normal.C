#include "fftlib.h"

/* 
 * This is on the "source" side.
 * there are srcSize[0] rows of size srcSize[1] each, in row-major order
 */
void
NormalRealSlabArray::doFFT(int src_id, int dst_id)
{
    NormalFFTinfo &fftinfo = *(fftinfos[src_id]);
    
    if(fftinfo.transformType != REAL_TO_COMPLEX) CkPrintf("Transform Type at doFFT is %d\n", fftinfo.transformType);
    CkAssert(fftinfo.transformType == REAL_TO_COMPLEX);
    
    int rplaneSize = fftinfo.srcSize[0] * fftinfo.srcSize[1];
    int cplaneSize = (fftinfo.srcSize[0]/2+1) * fftinfo.srcSize[1];
    int lineSize = fftinfo.srcSize[1];

    double *dataPtr = (double*)fftinfo.dataPtr;
    complex *outData = new complex[cplaneSize * fftinfo.srcPlanesPerSlab];
    complex *outDataPtr = outData;

	// do the 2D forward ffts
    CmiAssert(rfwd1DXPlan!=NULL && fwd1DYPlan!=NULL);
    int p;

    for(p = 0; p < fftinfo.srcPlanesPerSlab; p++){
        // real_to_complex on X dimension
	rfftw(rfwd1DXPlan, fftinfo.srcSize[1], 
	      (fftw_real*)dataPtr, fftinfo.srcSize[0], 1, 
	      (fftw_real*)outDataPtr, fftinfo.srcSize[0]/2+1, 1);
	fftw(fwd1DYPlan, fftinfo.srcSize[0]/2+1, 
	     (fftw_complex*)outDataPtr, fftinfo.srcSize[0]/2+1, 1,
	     NULL, 0, 0);
    }

    // allocating the data for sending to destination side
    complex *sendData = new complex[fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab * lineSize];
    complex *temp;
    // i <> pe if destPlanesPerSlab is not 1
    int pe, i;
    for(i = 0, pe = 0; i < fftinfo.srcSize[0]/2+1; i += fftinfo.destPlanesPerSlab, pe++) {
	int ti;
	temp = sendData;
	for(ti = i; ti < i + fftinfo.destPlanesPerSlab && ti < fftinfo.srcSize[0]/2+1; ti++)
	    for(p = 0; p < fftinfo.srcPlanesPerSlab; p++) {
		memcpy(temp, 
		       outData + p * cplaneSize + ti * lineSize,
		       sizeof(complex) * lineSize);
		temp += lineSize;
	    }
	
//	CkPrintf("[%d] Sending out to acceptFFT [%d]\n", thisIndex, pe);
	((CProxy_NormalRealSlabArray)fftinfo.destProxy)(pe).acceptDataForFFT(lineSize * fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab, sendData, thisIndex, dst_id);
    }
    delete [] sendData;
    delete [] outData;
}

/*
 * This is on the "destination" side.
 */
void
NormalRealSlabArray::acceptDataForFFT(int numPoints, complex *points, int posn, int info_id)
{
    NormalFFTinfo &fftinfo = *(fftinfos[info_id]);
    CkAssert(fftinfo.transformType == COMPLEX_TO_REAL);

    complex *dataPtr = (complex*)fftinfo.dataPtr;
    int lineSize = fftinfo.destSize[1];
    
#if CAREFUL
    CkAssert(numPoints == fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab * lineSize);
#endif
	
    counts[info_id]++;
    int planeSize = fftinfo.destSize[0] * fftinfo.destSize[1];
    int p;
    for (p = 0; p < fftinfo.destPlanesPerSlab; p++) {
	memcpy(dataPtr + posn * fftinfo.srcPlanesPerSlab * lineSize + p * planeSize,
	       points, 
	       sizeof(complex) * lineSize * fftinfo.srcPlanesPerSlab);
	points += lineSize * fftinfo.srcPlanesPerSlab;
    }
    if (counts[info_id] == fftinfo.destSize[0] / fftinfo.srcPlanesPerSlab) {
	counts[info_id] = 0;
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
    NormalFFTinfo &fftinfo = *(fftinfos[src_id]);
    CkAssert(fftinfo.transformType == COMPLEX_TO_REAL);

    complex *dataPtr = (complex*)fftinfo.dataPtr;
    int planeSize = fftinfo.destSize[0] * fftinfo.destSize[1];
    int lineSize = fftinfo.destSize[1];
    
    CmiAssert(bwd1DZPlan!=NULL && bwd1DYPlan!=NULL);
    int p;

    for(p = 0; p < fftinfo.destPlanesPerSlab; p++){
	fftw(bwd1DZPlan, fftinfo.destSize[0], 
	     (fftw_complex*)dataPtr+p*planeSize, fftinfo.destSize[0], 1, 
	     NULL, 0, 0);
	fftw(bwd1DYPlan, fftinfo.destSize[1], 
	     (fftw_complex*)dataPtr+p*planeSize, 1, fftinfo.destSize[1], 
	     NULL, 0, 0);
    }
  
    complex *sendData = new complex[fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab * lineSize];
    complex *temp;
    int i, pe;
    for (i = 0, pe = 0; i < fftinfo.destSize[0]; i += fftinfo.srcPlanesPerSlab, pe++) {
	int ti;
	temp = sendData;
	for (ti = i; ti < i + fftinfo.srcPlanesPerSlab && ti < fftinfo.destSize[0]/2+1; ti++)
	    for (p = 0; p < fftinfo.destPlanesPerSlab; p++) {
		memcpy(temp,
		       dataPtr + p * planeSize + ti * lineSize,
		       sizeof(complex) * lineSize);
		temp += lineSize;
	    }
//	CkPrintf("[%d] Sending out to acceptIFFT [%d]\n", thisIndex, pe);
	((CProxy_NormalRealSlabArray)fftinfo.srcProxy)(pe).acceptDataForIFFT(lineSize * fftinfo.destPlanesPerSlab * fftinfo.srcPlanesPerSlab, sendData, thisIndex, dst_id);
    }
    delete [] sendData;
}

void
NormalRealSlabArray::acceptDataForIFFT(int numPoints, complex *points, int posn, int info_id)
{
    NormalFFTinfo &fftinfo = *(fftinfos[info_id]);
    CkAssert(fftinfo.transformType == REAL_TO_COMPLEX);

    int rplaneSize = fftinfo.srcSize[0] * fftinfo.srcSize[1];
    int cplaneSize = (fftinfo.srcSize[0]/2+1) * fftinfo.srcSize[1];
    int planeSize = fftinfo.destSize[0] * fftinfo.destSize[1];
    int lineSize = fftinfo.destSize[1];
    double *dataPtr = (double*)fftinfo.dataPtr;
    complex *inData = new complex[fftinfo.srcPlanesPerSlab * cplaneSize];
#if CAREFUL
    CkAssert(numPoints == fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab * lineSize);
#endif
    
    counts[info_id]++;
    int p;
    for(p = 0; p < fftinfo.srcPlanesPerSlab; p++) {
	memcpy(dataPtr + p * planeSize + posn * lineSize * fftinfo.destPlanesPerSlab,
	       points, 
	       sizeof(complex) * lineSize * fftinfo.destPlanesPerSlab);
	points += lineSize * fftinfo.destPlanesPerSlab;
    }
    
    int expectedCount = (fftinfo.srcSize[0]/2+1)%fftinfo.destPlanesPerSlab==0 ? 
	(fftinfo.srcSize[0]/2+1) / fftinfo.destPlanesPerSlab : (fftinfo.srcSize[0]/2+1) / fftinfo.destPlanesPerSlab+1;
    if (counts[info_id] == expectedCount) {
	counts[info_id] = 0;
	CmiAssert(rbwd1DXPlan!=NULL);
	for(p = 0; p < fftinfo.srcPlanesPerSlab; p++) {
	    // complex_to_real transform
		rfftw(rbwd1DXPlan,
		      lineSize,
		      (fftw_real*)inData + p * cplaneSize,
		      lineSize, 1, //stride, nextFFT
		      (fftw_real*)dataPtr + p * planeSize, lineSize, 1);
	}
	doneIFFT(info_id);
    }
    delete [] inData;
}
void NormalRealSlabArray::createPlans(NormalFFTinfo &info)
{
    if (info.isSrcSlab) {
	rfwd1DXPlan = rfftw_create_plan(info.srcSize[0], FFTW_REAL_TO_COMPLEX, FFTW_MEASURE|FFTW_OUT_OF_PLACE);
	fwd1DYPlan = fftw_create_plan(info.srcSize[1], FFTW_BACKWARD, FFTW_MEASURE|FFTW_IN_PLACE); 
	rbwd1DXPlan = rfftw_create_plan(info.srcSize[0], FFTW_BACKWARD, FFTW_MEASURE|FFTW_OUT_OF_PLACE);
    }
    else {
	bwd1DZPlan = fftw_create_plan(info.destSize[0], FFTW_COMPLEX_TO_REAL, FFTW_MEASURE|FFTW_IN_PLACE);
	bwd1DYPlan = fftw_create_plan(info.destSize[1], FFTW_BACKWARD, FFTW_MEASURE|FFTW_IN_PLACE);
	fwd1DZPlan = fftw_create_plan(info.destSize[0], FFTW_FORWARD, FFTW_MEASURE|FFTW_IN_PLACE);
    }
}

void NormalRealSlabArray::setup(NormalFFTinfo &info)
{
    fftinfos[0] = new NormalFFTinfo(info);
    counts[0] = 0;
    rfwd1DXPlan = rbwd1DXPlan = (rfftw_plan) NULL;
    fwd1DYPlan = bwd1DYPlan = (fftw_plan) NULL;
    fwd1DZPlan = bwd1DZPlan = (fftw_plan) NULL;

    createPlans(info);
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
    int i;
    for (i = 0; i < MAX_FFTS; i++)
	delete fftinfos[i];
}


void NormalRealSlabArray::pup(PUP::er &p)
{
    int i;
    ArrayElement1D::pup(p);

    for (i = 0; i < MAX_FFTS; i++) {
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
    
    p(counts, MAX_FFTS);
}


