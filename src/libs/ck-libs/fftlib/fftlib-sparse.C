#include "fftlib.h"

/*
 * organization of data: 
 * On the "source" side -
 * there are srcSize[0] rows of size srcSize[1] each. 
 */
#if 0
void
SlabArray::doFFT()
{
    // pointer to region of data; this may be in run-length form.
    complex *srcData = data();
    FFTinfo *infoPtr = getFFTinfo();
    const CkIndex2D *idx = myIdx();
    complex *dataPtr = 0;
    if (info->packingMode == SPARSE_MODE) {
		// get the run-length information
		RunDescriptor *runs;
		int numRuns, numPoints;
		getRuns(&runs, &numRuns, &numPoints);
		dataPtr = new complex[infoPtr->srcSize[0] * infoPtr->srcSize[1] * infoPtr->srcPlanesPerSlab];
		int i, l, numCovered = 0;
		int planeSize = info->srcSize[0] * info->srcSize[1];
		bool *nonZero = new int[infoPtr->srcSize[1] * infoPtr->srcPlanesPerSlab];
		int numNonZero = 0;
		for (i = 0; i < infoPtr->srcSize[1] * infoPtr->srcPlanesPerSlab; i++)
			nonZero[i] = false;
		for (r = 0; r < numRuns; r++) {
			// the starting position of the run in the data array (dataPtr)
			int pos = runs[r].x * planeSize + runs[r].y * info->srcSize[1];

			// copying the data into the actual array where the fft will be
			// done.
			for (l = 0; l < runs[r].length; l++) {
				dataPtr[pos + runs[r].offset(l)] = srcData[numCovered++];
				nonZero[runs[r].x * infoPtr->srcSize[1] + runs[r].offset(l)] = true;
				numNonZero++;
			}
		}
		// Here we do only a one-dimensional fft.
		int p;
		for(p = 0; p < srcPlanesPerSlab; p++)
			for (z = 0; z < info->srcSize[1]; z++)
				if (nonZero[p * info->srcSize[1] + z])
					fftw_one(fwd1DPlan, // the forward plan
							 1,
							 dataPtr + p * infoPtr->srcSize[0] * infoPtr->srcSize[1] + z, // the pointer to the data
							 infoPtr->srcSize[0], 1,
							 NULL, 0, 0);

		// Now, to send the data: we send only the non-zero parts.
		// This is the advantage of doing the 1D ffts on the source side.
		// this is done by augmenting the data with an array of integer flags
		// The array does not change within the context of a single plane
		complex *sendData = new complex[numNonZero * infoPtr->destPlanesPerSlab];
		int z, y, numCovered;
		for (y = 0; y < infoPtr->srcSize[0]; y += infoPtr->destPlanesPerSlab) {
			int t;
			numCovered = 0;
			for (t = y; t < y + infoPtr->destPlanesPerSlab; t++) 
				for (p = 0; p < infoPtr->srcPlanesPerSlab; p++) 
					for (z = 0; z < infoPtr->srcSize[1]; z++)
						if (nonZero[p * infoPtr->srcSize[1] + z]) {
							sendData[numCovered] = dataPtr[p * infoPtr->srcSize[0] * infoPtr->srcSize[1] + t * infoPtr->srcSize[1] + z];
							numCovered++;
						}
			
			destProxy(idx->x, y).acceptDataForFFT(infoPtr->srcSize[1] * infoPtr->srcPlanesPerSlab, numNonZero * infoPtr->destPlanesPerSlab, sendData, idx->y);
		}

		delete [] nonZero;
		delete [] sendData;
		return;
    }
    
	
	// The following part handles the case where packingMode equals DENSE_MODE
	
	// Do the fft in the planes
    int p;
    for(p = 0; p < srcPlanesPerSlab; p++)
	fftwnd_one(fwd2DPlan, 
		   dataPtr + p * infoPtr->srcSize[0] * infoPtr->srcSize[1],
		   NULL);
    
    // allocating the data for sending to destination side
    complex *sendData = new complex[srcPlanesPerSlab * destPlanesPerSlab * infoPtr->srcSize[1]];
    complex *temp;
    int i;
    for(i = 0; i < infoPtr->srcSize[0]; i += destPlanesPerSlab) {
		int ti;
		temp = sendData;
		for(ti = i; ti < i + destPlanesPerSlab; ti++)
			for(p = 0; p < srcPlanesPerSlab; p++) {
				memcpy(temp, 
					   dataPtr + p * infoPtr->srcSize[0] * infoPtr->srcSize[1] + ti * infoPtr->srcSize[1],
					   sizeof(complex) * infoPtr->srcSize[1]);
				temp += infoPtr->srcSize[1];
			}
	
		destProxy(idx->x, i).acceptDataForFFT(0, 0, infoPtr->srcSize[1] * srcPlanesPerSlab * destPlanesPerSlab, sendData, idx->y);
    }
    delete [] sendData;
    delete [] dataPtr;
}

void
SlabArray::acceptDataForFFT(int n, int *flags, int dataSize, complex *data, int posn)
{
    complex *dataPtr = this->data();
	FFTinfo *infoPtr = getFFTinfo();    

    count++;
	if (infoPtr->packingMode == SPARSE_MODE) {
		CkAssert (n && flags);

		//first, copy the flags in the right place
		memcpy (nonZero + posn * infoPtr->destSize[], flags, sizeof(int) * n);
	}
	else {
		int p;
		for (p = 0; p < infoPtr->destPlanesPerSlab; p++) {
			memcpy(dataPtr + posn * infoPtr->destSize[1] + p * infoPtr->destSize[0] * infoPtr->destSize[1],
				   data, 
				   sizeof(complex) * infoPtr->srcPlanesPerSlab * infoPtr->destSize[1]);
			data += destSize2 * srcPlanesPerSlab;
		}
		if (count == info->destSize[0] / srcPlanesPerSlab) {
			count = 0;
			for(p = 0; p < destPlanesPerSlab; p++)
				fftw(fwd1DPlan, 
					 destSize2,
					 dataPtr + p * destSize1 * destSize2,
					 destSize2, 1,
					 NULL, 0, 0);
	
			doneFFT();
		}
	}
}

void
SlabArray::doIFFT()
{
    complex *dataPtr = data();
    const CkIndex2D *idx = myIdx();
    
    int p;
    for(p = 0; p < destPlanesPerSlab; p++)
	fftwnd_one(bwd2DPlan, 
		   dataPtr + p * destSize1 * destSize2,
		   NULL);
    
    complex *sendData = new complex[srcPlanesPerSlab * destPlanesPerSlab * destSize2];
    complex *temp;
    int i;
    for (i = 0; i < destSize1; i += srcPlanesPerSlab) {
	int ti;
	temp = sendData;
	for (ti = i; ti < i + srcPlanesPerSlab; ti++)
	    for (p = 0; p < destPlanesPerSlab; p++) {
		memcpy(temp,
		       dataPtr + p * destSize1 * destSize2 + ti * destSize2,
		       sizeof(complex) * destSize2);
		temp += destSize2;
	    }
	srcProxy(idx->x, i).acceptDataForIFFT(destSize2 * destPlanesPerSlab * srcPlanesPerSlab, sendData, idx->y);
    }
    delete [] sendData;
}

void
SlabArray::acceptDataForIFFT(int dataSize, complex *data, int posn)
{
    complex *dataPtr = this->data();
    
    count++;
    int p;
    for(p = 0; p < srcPlanesPerSlab; p++) {
		memcpy(dataPtr + p * infoPtr->srcSize[0] * infoPtr->srcSize[1] + posn * infoPtr->srcSize[1],
			   data , 
			   sizeof(complex) * infoPtr->srcSize[1] * destPlanesPerSlab);
		data += destPlanesPerSlab * infoPtr->srcSize[1];
    }
    
    if (count == infoPtr->srcSize[0] / destPlanesPerSlab) {
		count = 0;
		for(p = 0; p < srcPlanesPerSlab; p++)
			fftw(bwd1DPlan,
				 infoPtr->srcSize[1],
				 dataPtr + p * infoPtr->srcSize[0] * infoPtr->srcSize[1],
				 infoPtr->srcSize[1], 1,
				 NULL, 0, 0);
		doneIFFT();
    }
}

SlabArray::SlabArray(FFTinfo &info) 
{
    count = 0;
    if (info.isSrcSlab) {
		if (info.packingMode == SPARSE_MODE) {
			fwd1DPlan = fftw_create_plan(info.srcSize[0], FFTW_FORWARD, FFTW_USE_WISDOM|FFTW_MEASURE|FFTW_IN_PLACE);
			bwd1DPlan = fftw_create_plan(info.srcSize[0], FFTW_BACKWARD, FFTW_USE_WISDOM|FFTW_MEASURE|FFTW_IN_PLACE);
			nonZero = new int[info.destSize[0] * info.destSize[1]];
		}
		else {
			fwd2DPlan = fftw2d_create_plan(info.srcSize[0], info.srcSize[1], FFTW_FORWARD, FFTW_USE_WISDOM|FFTW_MEASURE|FFTW_IN_PLACE);
			bwd1DPlan = fftw_create_plan(info.srcSize[1], FFTW_BACKWARD, FFTW_USE_WISDOM|FFTW_MEASURE|FFTW_IN_PLACE);
		}
    }
    else {
		if (info.packingMode == SPARSE_MODE) {
			fwd1DaPlan = fftw_create_plan(info.destSize[], FFTW_FORWARD, FFTW_USE_WISDOM|FFTW_MEASURE|FFTW_IN_PLACE);
			fwd1DbPlan = fftw_create_plan(info.destSize[], FFTW_FORWARD, FFTW_USE_WISDOM|FFTW_MEASURE|FFTW_IN_PLACE);
			bwd1DaPlan = fftw_create_plan(info.destSize[], FFTW_BACKWARD, FFTW_USE_WISDOM|FFTW_MEASURE|FFTW_IN_PLACE);
			bwd1DbPlan = fftw_create_plan(info.destSize[], FFTW_BACKWARD, FFTW_USE_WISDOM|FFTW_MEASURE|FFTW_IN_PLACE);
			fwd2DPlan = fftw2d_create_plan(info.destSize[0], info.destSize[1], FFTW_FORWARD, FFTW_USE_WISDOM|FFTW_MEASURE|FFTW_IN_PLACE);
			bwd2DPlan = fftw2d_create_plan(info.destSize[0], info.destSize[1], FFTW_BACWARD, FFTW_USE_WISDOM|FFTW_MEASURE|FFTW_IN_PLACE);
		}
		else {
			bwd2DPlan = fftw2d_create_plan(info.destSize[0], info.destSize[1], FFTW_BACKWARD, FFTW_USE_WISDOM|FFTW_MEASURE|FFTW_IN_PLACE);
			fwd1DPlan = fftw_create_plan(info.destSize[1], FFTW_FORWARD, FFTW_USE_WISDOM|FFTW_MEASURE|FFTW_IN_PLACE);
		}
    }
}

#endif
