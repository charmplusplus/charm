#include "fftlib.h"
/* 
 * This is on the "source" side.
 * there are srcSize[0] rows of size srcSize[1] each, in row-major order
 */
void
NormalSlabArray::doFFT(int src_id, int dst_id)
{
#if VERBOSE
  CkPrintf("[%d] doFFT\n",thisIndex);  
#endif 
    NormalFFTinfo &fftinfo = (infoVec[src_id]->info);

    CkAssert(fftinfo.transformType == COMPLEX_TO_COMPLEX);

    ckcomplex *dataPtr = (ckcomplex*)fftinfo.dataPtr;

    int planeSize = fftinfo.srcSize[0] * fftinfo.srcSize[1];
    int lineSize = fftinfo.srcSize[1];

    CmiAssert(fwd2DPlan!=NULL);

    int p;
    for(p = 0; p < fftinfo.srcPlanesPerSlab; p++){
	FFT2_ONE(fwd2DPlan, (dataPtr + p * planeSize));
    }
    // allocating the data for sending to destination side
    ckcomplex *sendData = new ckcomplex[fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab * lineSize];
    ckcomplex *temp;

    CProxy_NormalSlabArray destProxy_com;
    ComlibInstanceHandle fftcommInstance = (infoVec[src_id]->fftcommInstance);
    if (fftuseCommlib) {
	if(fftinfo.isSrcSlab)
	    destProxy_com = (CProxy_NormalSlabArray)destProxy;
	else
	    destProxy_com = (CProxy_NormalSlabArray)srcProxy;
	ComlibAssociateProxy(&fftcommInstance, destProxy_com);
	//ComlibBeginIteration(destProxy_com);
	fftcommInstance.beginIteration();
    }

    int pe, i;
    for(i = 0, pe = 0; i < fftinfo.srcSize[0]; i += fftinfo.destPlanesPerSlab, pe++) {
	int ti;
	temp = sendData;
	for(ti = i; ti < i + fftinfo.destPlanesPerSlab; ti++)
	    for(p = 0; p < fftinfo.srcPlanesPerSlab; p++) {
		memcpy(temp, 
		       dataPtr + p * planeSize + ti * lineSize,
		       sizeof(ckcomplex) * lineSize);
		temp += lineSize;
	    }
	if (fftuseCommlib)	
	((CProxy_NormalSlabArray)destProxy_com)(pe).acceptDataForFFT(lineSize * fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab, sendData, thisIndex, dst_id);
	else
	((CProxy_NormalSlabArray)destProxy)(pe).acceptDataForFFT(lineSize * fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab, sendData, thisIndex, dst_id);
    }
    if (fftuseCommlib) {
//	ComlibEndIteration(destProxy_com);
	fftcommInstance.endIteration();
    }
    delete [] sendData;
}

/*
 * This is on the "destination" side.
 */
void
NormalSlabArray::acceptDataForFFT(int numPoints,ckcomplex  *points, int posn, int info_id)
{
#if VERBOSE
  CkPrintf("[%d] acceptDataFFT\n",thisIndex);
#endif

    NormalFFTinfo &fftinfo = (infoVec[info_id]->info);

    CkAssert(fftinfo.transformType == COMPLEX_TO_COMPLEX);

   ckcomplex  *dataPtr = (ckcomplex*)fftinfo.dataPtr;
    int lineSize = fftinfo.destSize[1];
    
#if CAREFUL
    CkAssert(numPoints == fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab * lineSize);
#endif
	
    infoVec[info_id]->count++;
    int planeSize = fftinfo.destSize[0] * fftinfo.destSize[1];
    int p;
    for (p = 0; p < fftinfo.destPlanesPerSlab; p++) {
	memcpy(dataPtr + posn * fftinfo.srcPlanesPerSlab * lineSize + p * planeSize,
	       points, 
	       sizeof(ckcomplex) * lineSize * fftinfo.srcPlanesPerSlab);
	points += lineSize * fftinfo.srcPlanesPerSlab;
    }
    if (infoVec[info_id]->count == fftinfo.destSize[0] / fftinfo.srcPlanesPerSlab) {
	infoVec[info_id]->count = 0;
	CkAssert(fwd1DPlan != NULL);
	for(p = 0; p < fftinfo.destPlanesPerSlab; p++) {

	  //	  FFT1_MANY(fwd1DPlan,lineSize,(dataPtr + p * planeSize));
	  FFT1_MANY(fwd1DPlan,lineSize,(dataPtr + p * planeSize));
#ifdef CMK_VERSION_BLUEGENE
		CmiNetworkProgress();
#endif
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
#if VERBOSE
  CkPrintf("[%d] doIFFT\n",thisIndex);
#endif
    NormalFFTinfo &fftinfo = (infoVec[src_id]->info);

    CkAssert(fftinfo.transformType == COMPLEX_TO_COMPLEX);

   ckcomplex  *dataPtr = (ckcomplex*)fftinfo.dataPtr;
    int planeSize = fftinfo.destSize[0] * fftinfo.destSize[1];
    int lineSize = fftinfo.destSize[1];
    
    CmiAssert(bwd2DPlan!=NULL);
    int p;

    for(p = 0; p < fftinfo.destPlanesPerSlab; p++){
	FFT2_ONE(bwd2DPlan,dataPtr + p * planeSize);
	#if VERBOSE
	if(thisIndex==0 && src_id==0){
	    for(int i=0;i<fftinfo.srcSize[1]*fftinfo.srcSize[0]; i++)
		CkPrintf("%d %g %g\n", i, dataPtr[p * planeSize+i].re,  dataPtr[p * planeSize+i].im);
	}
	#endif
    }
    
    CProxy_NormalSlabArray srcProxy_com;
    ComlibInstanceHandle fftcommInstance = (infoVec[src_id]->fftcommInstance);
    if (fftuseCommlib) {
	if(fftinfo.isSrcSlab)
	    srcProxy_com = (CProxy_NormalSlabArray)destProxy;
	else
	    srcProxy_com = (CProxy_NormalSlabArray)srcProxy;
	ComlibAssociateProxy(&fftcommInstance, srcProxy_com);
	//ComlibBeginIteration(destProxy_com);
	fftcommInstance.beginIteration();
    }

   ckcomplex  *sendData = new ckcomplex[fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab * lineSize];
   ckcomplex  *temp;
    int i, pe;
    for (i = 0, pe = 0; i < fftinfo.destSize[0]; i += fftinfo.srcPlanesPerSlab, pe++) {
	int ti;

	temp = sendData;
	for (ti = i; ti < i + fftinfo.srcPlanesPerSlab; ti++)
	    for (p = 0; p < fftinfo.destPlanesPerSlab; p++) {
		memcpy(temp,
		       dataPtr + p * planeSize + ti * lineSize,
		       sizeof(ckcomplex) * lineSize);
		temp += lineSize;
	    }
    if (fftuseCommlib)
	((CProxy_NormalSlabArray)srcProxy_com)(pe).acceptDataForIFFT(lineSize * fftinfo.destPlanesPerSlab * fftinfo.srcPlanesPerSlab, sendData, thisIndex, dst_id);
    else
	((CProxy_NormalSlabArray)srcProxy)(pe).acceptDataForIFFT(lineSize * fftinfo.destPlanesPerSlab * fftinfo.srcPlanesPerSlab, sendData, thisIndex, dst_id);

    }
    if (fftuseCommlib) {
//	ComlibEndIteration(srcProxy_com);
	fftcommInstance.endIteration();
    }
    delete [] sendData;
}


void
NormalSlabArray::acceptDataForIFFT(int numPoints,ckcomplex  *points, int posn, int info_id)
{
#if VERBOSE
  CkPrintf("[%d] acceptIFFT\n",thisIndex);
#endif

    NormalFFTinfo &fftinfo = (infoVec[info_id]->info);

    CkAssert(fftinfo.transformType == COMPLEX_TO_COMPLEX);

   ckcomplex  *dataPtr = (ckcomplex*)fftinfo.dataPtr;
    int planeSize = fftinfo.srcSize[0] * fftinfo.srcSize[1];
    int lineSize = fftinfo.srcSize[1];
#if CAREFUL
    CkAssert(numPoints == fftinfo.srcPlanesPerSlab * fftinfo.destPlanesPerSlab * lineSize);
#endif

    infoVec[info_id]->count++;
    int p;
    for(p = 0; p < fftinfo.srcPlanesPerSlab; p++) {
	memcpy(dataPtr + p * planeSize + posn * lineSize * fftinfo.destPlanesPerSlab,
	       points, 
	       sizeof(ckcomplex) * lineSize * fftinfo.destPlanesPerSlab);
	points += lineSize * fftinfo.destPlanesPerSlab;
    }
    
    if (infoVec[info_id]->count == fftinfo.srcSize[0] / fftinfo.destPlanesPerSlab) {
	infoVec[info_id]->count = 0;
	CmiAssert(bwd1DPlan!=NULL);
	for(p = 0; p < fftinfo.srcPlanesPerSlab; p++) {
	  FFT1_MANY(bwd1DPlan,lineSize, dataPtr + p * planeSize);
#ifdef CMK_VERSION_BLUEGENE
		CmiNetworkProgress();
#endif

	}

// Refactoring , so that FFT then IFFT would give us the original data
	double factor = fftinfo.srcSize[0] * fftinfo.srcSize[1] * fftinfo.destSize[0];
	for(p = 0; p < fftinfo.srcPlanesPerSlab; p++) {
	   ckcomplex  *tempdataPtr = dataPtr + p * planeSize;
	    for(int i = 0; i<fftinfo.destSize[0]*fftinfo.destSize[1]; i++){
		tempdataPtr[i].re/= factor;
		tempdataPtr[i].im/= factor;
	    }
	}

	doneIFFT(info_id);
    }
}

void NormalSlabArray::createPlans(NormalFFTinfo &info)
{
#if FFT_USE_FFTW_2_1_5	
    if (info.isSrcSlab) {
	fwd2DPlan = fftw2d_create_plan(info.srcSize[0], info.srcSize[1], FFTW_FORWARD, FFTW_IN_PLACE);
	bwd1DPlan = fftw_create_plan(info.srcSize[0], FFTW_BACKWARD, FFTW_IN_PLACE);
    }
    else {
	bwd2DPlan = fftw2d_create_plan(info.destSize[0], info.destSize[1], FFTW_BACKWARD, FFTW_IN_PLACE);
	fwd1DPlan = fftw_create_plan(info.destSize[0], FFTW_FORWARD, FFTW_IN_PLACE);
    }
#elif FFT_USE_FFTW_3_1

#elif FFT_USE_ESSL
    if(info.isSrcSlab){
	fwd2DPlan = essl_create_2Dplan_one(1, info.srcSize[1], info.srcSize[0], info.srcSize[1], 1);
	bwd1DPlan = essl_create_1Dplan_many(info.srcSize[1], 1, info.srcSize[0], info.srcSize[1], -1);    
    }
    else {
	bwd2DPlan = essl_create_2Dplan_one(1, info.destSize[1], info.destSize[0], info.destSize[1], -1);
	fwd1DPlan = essl_create_1Dplan_many(info.destSize[1], 1, info.destSize[0], info.destSize[1], 1);    
    }
#else 
    CkAbort("FFT NOT SUPPORTED YET FOR THIS OPTION!\n");
#endif
}

void NormalSlabArray::setup(NormalFFTinfo &info, 
			    CProxy_NormalSlabArray src, 
			    CProxy_NormalSlabArray dest,
			    bool _useCommlib, ComlibInstanceHandle inst)
{
    SlabArrayInfo *slabinfo = new SlabArrayInfo();
    
    slabinfo->info = info;
    slabinfo->count = 0;
    slabinfo->fftcommInstance = inst;
    infoVec.insert(infoVec.size(), slabinfo);

    srcProxy = src;
    destProxy = dest;

    if((info.isSrcSlab && fwd2DPlan==NULL) || (!info.isSrcSlab && bwd2DPlan==NULL))
	createPlans(info);

    fftuseCommlib = _useCommlib;
/*    fftcommInstance = ComlibInstanceHandle();
    if (fftuseCommlib) {        
	fftcommInstance = inst;
	if(info.isSrcSlab)
	    ComlibAssociateProxy(&inst, destProxy); 
	else
	    ComlibAssociateProxy(&inst, srcProxy); 
   }
*/
}

NormalSlabArray::NormalSlabArray(NormalFFTinfo &info, 
				 CProxy_NormalSlabArray src, 
				 CProxy_NormalSlabArray dest, 
				 bool useCommlib, 
				 ComlibInstanceHandle inst)  {
    NormalSlabArray();
    setup(info, src, dest, useCommlib, inst); 
}

NormalSlabArray::~NormalSlabArray() 
{
#if FFT_USE_FFTW_2_1_5
    if (fwd2DPlan)
	fftwnd_destroy_plan(fwd2DPlan);
    if (bwd2DPlan)
	fftwnd_destroy_plan(bwd2DPlan);
    if (fwd1DPlan)
	fftw_destroy_plan(fwd1DPlan);
    if (bwd1DPlan)
	fftw_destroy_plan(bwd1DPlan);
#elif FFT_USE_FFTW_3_1

#elif FFT_USE_ESSL
    if (fwd2DPlan)
	essl_destroy_plan(fwd2DPlan);
    if (bwd2DPlan)
	essl_destroy_plan(bwd2DPlan);
    if (fwd1DPlan)
	essl_destroy_plan(fwd1DPlan);
    if (bwd1DPlan)
	essl_destroy_plan(bwd1DPlan);
#endif
    infoVec.removeAll();
}


void NormalSlabArray::pup(PUP::er &p)
{
    ArrayElement1D::pup(p);
    //FIXME: PUP for info and plans
/*
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
*/
    p | fftuseCommlib;
    // p | fftcommInstance;
}


#include "fftlib.def.h"
