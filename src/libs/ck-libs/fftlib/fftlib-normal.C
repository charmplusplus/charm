#include "fftlib.h"

/* 
 * This is on the "source" side.
 * there are srcSize[0] rows of size srcSize[1] each, in row-major order
 */
void
NormalSlabArray::doFFT(int src_id, int dst_id)
{
    NormalFFTinfo &fftinfo = *(fftinfos[src_id]);

    if(fftinfo.transformType != COMPLEX_TO_COMPLEX) CkPrintf("Transform Type at doFFT is %d\n", fftinfo.transformType);
    CkAssert(fftinfo.transformType == COMPLEX_TO_COMPLEX);

    complex *dataPtr = (complex*)fftinfo.dataPtr;

    int planeSize = fftinfo.srcSize[0] * fftinfo.srcSize[1];
    int lineSize = fftinfo.srcSize[1];

	// do the 2D forward ffts
    CmiAssert(fwd2DPlan!=NULL);

    int p;
    for(p = 0; p < fftinfo.srcPlanesPerSlab; p++)
      fftwnd_one(fwd2DPlan, (fftw_complex*)dataPtr + p * planeSize, NULL);
    
    // allocating the data for sending to destination side
    complex *sendData = new complex[fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab * lineSize];
    complex *temp;
    // i <> pe if destPlanesPerSlab is not 1
    int pe, i;
    for(i = 0, pe = 0; i < fftinfo.srcSize[0]; i += fftinfo.destPlanesPerSlab, pe++) {
	int ti;
	temp = sendData;
	for(ti = i; ti < i + fftinfo.destPlanesPerSlab; ti++)
	    for(p = 0; p < fftinfo.srcPlanesPerSlab; p++) {
		memcpy(temp, 
		       dataPtr + p * planeSize + ti * lineSize,
		       sizeof(complex) * lineSize);
		temp += lineSize;
	    }
	
	((CProxy_NormalSlabArray)fftinfo.destProxy)(pe).acceptDataForFFT(lineSize * fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab, sendData, thisIndex, dst_id);
    }
    delete [] sendData;
}

/*
 * This is on the "destination" side.
 */
void
NormalSlabArray::acceptDataForFFT(int numPoints, complex *points, int posn, int info_id)
{
    NormalFFTinfo &fftinfo = *(fftinfos[info_id]);

    CkAssert(fftinfo.transformType == COMPLEX_TO_COMPLEX);

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
	CkAssert(fwd1DPlan != NULL);
	for(p = 0; p < fftinfo.destPlanesPerSlab; p++) {
		fftw(fwd1DPlan, 
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
 */
void
NormalSlabArray::doIFFT(int src_id, int dst_id)
{
    NormalFFTinfo &fftinfo = *(fftinfos[src_id]);

    if(fftinfo.transformType != COMPLEX_TO_COMPLEX) CkPrintf("Transform Type at doFFT is %d\n", fftinfo.transformType);
    CkAssert(fftinfo.transformType == COMPLEX_TO_COMPLEX);

    complex *dataPtr = (complex*)fftinfo.dataPtr;
    int planeSize = fftinfo.destSize[0] * fftinfo.destSize[1];
    int lineSize = fftinfo.destSize[1];
    
    CmiAssert(bwd2DPlan!=NULL);
    int p;
    for(p = 0; p < fftinfo.destPlanesPerSlab; p++)
      fftwnd_one(bwd2DPlan, 
	       (fftw_complex*)dataPtr + p * planeSize,
	       NULL);
    
    complex *sendData = new complex[fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab * lineSize];
    complex *temp;
    int i, pe;
    for (i = 0, pe = 0; i < fftinfo.destSize[0]; i += fftinfo.srcPlanesPerSlab, pe++) {
	int ti;
	temp = sendData;
	for (ti = i; ti < i + fftinfo.srcPlanesPerSlab; ti++)
	    for (p = 0; p < fftinfo.destPlanesPerSlab; p++) {
		memcpy(temp,
		       dataPtr + p * planeSize + ti * lineSize,
		       sizeof(complex) * lineSize);
		temp += lineSize;
	    }
	((CProxy_NormalSlabArray)fftinfo.srcProxy)(pe).acceptDataForIFFT(lineSize * fftinfo.destPlanesPerSlab * fftinfo.srcPlanesPerSlab, sendData, thisIndex, dst_id);
    }
    delete [] sendData;
}

void
NormalSlabArray::acceptDataForIFFT(int numPoints, complex *points, int posn, int info_id)
{
    NormalFFTinfo &fftinfo = *(fftinfos[info_id]);

    CkAssert(fftinfo.transformType == COMPLEX_TO_COMPLEX);

    complex *dataPtr = (complex*)fftinfo.dataPtr;
    int planeSize = fftinfo.srcSize[0] * fftinfo.srcSize[1];
    int lineSize = fftinfo.srcSize[1];
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
    
    if (counts[info_id] == fftinfo.srcSize[0] / fftinfo.destPlanesPerSlab) {
	counts[info_id] = 0;
	CmiAssert(bwd1DPlan!=NULL);
	for(p = 0; p < fftinfo.srcPlanesPerSlab; p++) {
		fftw(bwd1DPlan,
		     lineSize,
		     (fftw_complex*)dataPtr + p * planeSize,
		     lineSize, 1, //stride, nextFFT
		     NULL, 0, 0);
	}
	doneIFFT(info_id);
    }
}

void NormalSlabArray::createPlans(NormalFFTinfo &info)
{
    if (info.isSrcSlab) {
	fwd2DPlan = fftw2d_create_plan(info.srcSize[0], info.srcSize[1], FFTW_FORWARD, FFTW_MEASURE|FFTW_IN_PLACE);
	bwd1DPlan = fftw_create_plan(info.srcSize[0], FFTW_BACKWARD, FFTW_MEASURE|FFTW_IN_PLACE);
    }
    else {
	bwd2DPlan = fftw2d_create_plan(info.destSize[0], info.destSize[1], FFTW_BACKWARD, FFTW_MEASURE|FFTW_IN_PLACE);
	fwd1DPlan = fftw_create_plan(info.destSize[0], FFTW_FORWARD, FFTW_MEASURE|FFTW_IN_PLACE);
    }
}

void NormalSlabArray::setup(NormalFFTinfo &info)
{
    fftinfos[0] = new NormalFFTinfo(info);
    counts[0] = 0;
    fwd1DPlan = bwd1DPlan = NULL;
    fwd2DPlan = bwd2DPlan = NULL;
    
    createPlans(info);
}


NormalSlabArray::~NormalSlabArray() 
{
    if (fwd2DPlan)
	fftwnd_destroy_plan(fwd2DPlan);
    if (bwd2DPlan)
	fftwnd_destroy_plan(bwd2DPlan);
    if (fwd1DPlan)
	fftw_destroy_plan(fwd1DPlan);
    if (bwd1DPlan)
	fftw_destroy_plan(bwd1DPlan);
    int i;
    for (i = 0; i < MAX_FFTS; i++)
	delete fftinfos[i];
}


void NormalSlabArray::pup(PUP::er &p)
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


#include "fftlib.def.h"
