#include "fftlib.h"

//#define VERBOSE 1

void
NormalLineArray::doFirstFFT(int fftid, int direction)
{
    LineFFTinfo &fftinfo = *(fftinfos[fftid]);        
    complex *line = fftinfo.dataPtr;
    int sizeX = fftinfo.sizeX;
    int sizeZ = fftinfo.sizeZ;
    int sPencilsPerSlab = fftinfo.sPencilsPerSlab;

    if(direction)
	fftw(fwdplan, sPencilsPerSlab, line, 1, sizeX, NULL, 0, 0); // sPencilsPerSlab many 1-D fft's 
    else
	fftw_one(bwdplan, line, NULL);
    
    // thisIndex.x is y-coord
    int x, y, z;
#ifdef VERBOSE
    CkPrintf("First FFT done at [%d %d] [%d %d]\n", thisIndex.x, thisIndex.y,sizeX,sizeZ);
#endif
#ifdef COMLIB
    CkAssert(id != -1);
    ckout << "xlines " << id << endl;
    mgrProxy.ckLocalBranch()->beginIteration(id);
#endif
    if(direction){
	int  sendDataSize = sPencilsPerSlab;
	complex *sendData = new complex[sendDataSize];
	for (x = 0; x < sizeX; x++) {
	    for(y = 0; y < sPencilsPerSlab; y++){
		sendData[y] = line[x+sizeX*y];
	    }
	    
#ifdef VERBOSE
	    CkPrintf(" [%d %d] sending to YLINES [%d %d] \n",  thisIndex.x, thisIndex.y, thisIndex.y, x);
#endif
	    fftinfo.ylinesProxy(thisIndex.y, x).doSecondFFT(thisIndex.x, sendData, sendDataSize, fftid, direction);
	}
	delete [] sendData;
    }
    else {
	for (z = 0; z < sizeZ; z++) {
#ifdef VERBOSE
	    CkPrintf(" [%d %d] sending to YLINES [%d %d] \n",  thisIndex.x, thisIndex.y, z, thisIndex.x);
#endif

	    fftinfo.ylinesProxy(z, thisIndex.x).doSecondFFT(thisIndex.y, &line[z], 1, fftid, direction);
	}
    }
#ifdef COMLIB
    mgrProxy.ckLocalBranch()->endIteration();
#endif
    
}

void
NormalLineArray::doSecondFFT(int ypos, complex *val, int datasize, int fftid, int direction)
{
    LineFFTinfo &fftinfo = *(fftinfos[fftid]);        
    complex *line = fftinfo.dataPtr;
    int sizeY = fftinfo.sizeY;
    int sPencilsPerSlab = fftinfo.sPencilsPerSlab;
    int	expectSize = 0;
     if(direction)
	expectSize = sPencilsPerSlab;
    else
	expectSize = 1;
    CkAssert(datasize == expectSize);
    memcpy(line+ypos, val, datasize*sizeof(complex));

    count[fftid] += datasize;
    if (count[fftid] == sizeY) {
	count[fftid] = 0;
	int y;
	if(direction)
	    fftw_one(fwdplan, line, NULL);
	else
	    fftw_one(bwdplan, line, NULL);
	
#ifdef VERBOSE
	CkPrintf("Second FFT done at [%d %d]\n", thisIndex.x, thisIndex.y);
#endif
		// thisIndex.y is x-coord
#ifdef COMLIB
	CkAssert(id != -1);
	ckout << "ylines " << id << " " << CkMyPe() << endl;
	mgrProxy.ckLocalBranch()->beginIteration(id);
#endif
	if(direction)
	    for (y = 0; y < sizeY; y++) {
#ifdef VERBOSE
		CkPrintf(" [%d %d] sending to ZLINES [%d %d] \n",  thisIndex.x, thisIndex.y, thisIndex.y, y);
#endif
		(fftinfo.zlinesProxy)(thisIndex.y, y).doThirdFFT(thisIndex.x, 0, &line[y], 1, fftid, direction);
	    }
	else {
	    for (y = 0; y < sizeY; y+=sPencilsPerSlab) {
#ifdef VERBOSE
		CkPrintf(" [%d %d] sending to ZLINES [%d %d] \n",  thisIndex.x, thisIndex.y, y, thisIndex.y);
#endif
		(fftinfo.xlinesProxy)(y, thisIndex.x).doThirdFFT(thisIndex.y, y, &line[y], sPencilsPerSlab, fftid, direction);
	    }
	}
#ifdef COMLIB
	mgrProxy.ckLocalBranch()->endIteration();
#endif
	}
}

void
NormalLineArray::doThirdFFT(int zpos, int ypos, complex *val, int datasize, int fftid, int direction)
{
    LineFFTinfo &fftinfo = *(fftinfos[fftid]);        
    complex *line = fftinfo.dataPtr;
    int sizeX = fftinfo.sizeX;
    int sizeZ = fftinfo.sizeZ;
    int sPencilsPerSlab = fftinfo.sPencilsPerSlab;
    int expectSize=0, offset=0, i; 
    if(direction){
	expectSize = 1;
	offset = zpos;
	line[zpos] = val[0];
    }
    else{
	expectSize = sPencilsPerSlab;
	for(i=0,offset=ypos; i<expectSize; i++,offset+=sizeX)
	    line[offset] = val[i]; //
    }
    CkAssert(datasize == expectSize);

    count[fftid] += expectSize;

    if (count[fftid] == sizeX* expectSize) {
	count[fftid] = 0;
	if(direction)
	    fftw_one(fwdplan, line, NULL);
	else
	    fftw(bwdplan, sPencilsPerSlab, line, 1, sizeX, NULL, 0, 0); // sPencilsPerSlab many 1-D fft's 
#ifdef VERBOSE
	CkPrintf("Third FFT done at [%d %d]\n", thisIndex.x, thisIndex.y);
#endif

	doneFFT(id, direction);
//	contribute(sizeof(int), &count, CkReduction::sum_int);
    }
}

void 
NormalLineArray::doneFFT(int id, int direction){
    CkPrintf("FFT finished \n");
}
