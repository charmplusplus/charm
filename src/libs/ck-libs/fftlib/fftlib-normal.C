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

    CkPrintf("useCommlib = %d\n", useCommlib);
    if (useCommlib) {
	CProxy_NormalSlabArray destProxy_com;
	if(fftinfo.isSrcSlab)
	    destProxy_com = (CProxy_NormalSlabArray)fftinfo.destProxy;
	else
	    destProxy_com = (CProxy_NormalSlabArray)fftinfo.srcProxy;
	commInstance.beginIteration();
	ComlibDelegateProxy(&destProxy_com);
    }

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
#ifdef IFFT_DUMP
    char ofile[80];
    snprintf(ofile,80,"prebwdfft.%d_%d_%d.out",thisIndex,src_id,dst_id);
    FILE *ifd=fopen(ofile,"w");
    // output each plane so we can compare them
    for(int i=0; i<fftinfo.destPlanesPerSlab;i++)
	for(int point=0;point<planeSize;point++)
	fprintf(ifd,"plane %d src %d dest %d point %d %.10g %.10g\n",i+thisIndex*fftinfo.destPlanesPerSlab,src_id,dst_id,point+i*planeSize+thisIndex*fftinfo.destPlanesPerSlab*planeSize,((complex *) dataPtr + point + i * planeSize )->re,((complex *) dataPtr + point + i * planeSize )->im);
    fclose(ifd);
#endif    
    for(p = 0; p < fftinfo.destPlanesPerSlab; p++)
      fftwnd_one(bwd2DPlan, 
	       (fftw_complex*)dataPtr + p * planeSize,
	       NULL);

#ifdef IFFT_DUMP
    ofile[80];
    snprintf(ofile,80,"fftlibSending.%d_%d_%d.out",thisIndex,src_id,dst_id);
    ifd=fopen(ofile,"w");
    // output each plane so we can compare them
    for(int i=0; i<fftinfo.destPlanesPerSlab;i++)
	for(int point=0;point<planeSize;point++)
	fprintf(ifd,"plane %d src %d dest %d point %d %.10g %.10g\n",i+thisIndex*fftinfo.destPlanesPerSlab,src_id,dst_id,point+i*planeSize,((complex *) dataPtr + point + i * planeSize )->re,((complex *) dataPtr + point + i * planeSize )->im);
    fclose(ifd);
#endif    
    
    if (useCommlib) {
	CProxy_NormalSlabArray destProxy_com;
	if(fftinfo.isSrcSlab)
	    destProxy_com = (CProxy_NormalSlabArray)fftinfo.destProxy;
	else
	    destProxy_com = (CProxy_NormalSlabArray)fftinfo.srcProxy;
	commInstance.beginIteration();
	ComlibDelegateProxy(&destProxy_com);
    }

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

    //back on source side
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
#ifdef IFFT_DUMP
    char ofile[80];
    CkPrintf("accept data for ifft info_id %d thisIndex %d posn %d\n",info_id,thisIndex, posn);
    snprintf(ofile,80,"fftlibaccepted%d_%d_%d.out",info_id,thisIndex,posn);
    FILE *ifd=fopen(ofile,"w");
#endif    
    for(p = 0; p < fftinfo.srcPlanesPerSlab; p++) {
	memcpy(dataPtr + p * planeSize + posn * lineSize * fftinfo.destPlanesPerSlab,
	       points, 
	       sizeof(complex) * lineSize * fftinfo.destPlanesPerSlab);
	points += lineSize * fftinfo.destPlanesPerSlab;
#ifdef IFFT_DUMP
	for(int apoint=0;apoint<lineSize*fftinfo.destPlanesPerSlab;apoint++)
	  fprintf(ifd,"p %d r %.10g i %.10g\n", p,((complex *) dataPtr +p *planeSize +posn*lineSize*fftinfo.destPlanesPerSlab+apoint)->re,((complex *) dataPtr +p *planeSize +posn*lineSize*fftinfo.destPlanesPerSlab+apoint)->im);
#endif
    }
#ifdef IFFT_DUMP
    fclose(ifd);
#endif
    
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

void NormalSlabArray::setup(NormalFFTinfo &info, bool _useCommlib)
{
    fftinfos[0] = new NormalFFTinfo(info);
    counts[0] = 0;
    fwd1DPlan = bwd1DPlan = NULL;
    fwd2DPlan = bwd2DPlan = NULL;
    
    createPlans(info);

    useCommlib = _useCommlib;
    if (useCommlib) {        
	  int period_in_ms = 1, nmsgs = 1000;
	  StreamingStrategy * strat = new StreamingStrategy(period_in_ms, nmsgs);
        commInstance = CkGetComlibInstance();
        commInstance.setStrategy(strat);
    }
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
