#include "fftlib.h"

//#define VERBOSE 1

void
NormalLineArray::doFirstFFT(int fftid, int direction)
{
    LineFFTinfo &fftinfo = (infoVec[fftid]->info);
    complex *line = fftinfo.dataPtr;
    int sizeX = fftinfo.sizeX;
    int sizeZ = fftinfo.sizeZ;
    int xPencilsPerSlab = fftinfo.xPencilsPerSlab;
    int yPencilsPerSlab = fftinfo.yPencilsPerSlab;
    int zPencilsPerSlab = fftinfo.zPencilsPerSlab;

    if(direction)
	fftw(fwdplan, xPencilsPerSlab, (fftw_complex*)line, 1, sizeX, NULL, 0, 0); // xPencilsPerSlab many 1-D fft's 
    else
	fftw(bwdplan, zPencilsPerSlab, (fftw_complex*)line, 1, sizeZ, NULL, 0, 0);
    
    int x, y, z=0;
#ifdef VERBOSE
    CkPrintf("First FFT done at [%d %d] [%d %d]\n", thisIndex.x, thisIndex.y,sizeX,sizeZ);
#endif
#ifdef COMLIB
    CkAssert(id != -1);
    ckout << "xlines " << id << endl;
    mgrProxy.ckLocalBranch()->beginIteration(id);
#endif
    int baseY;
    if(direction)
    {
	int  sendDataSize = yPencilsPerSlab*xPencilsPerSlab;
	complex *sendData = new complex[sendDataSize];
	int zpos = thisIndex.y;
	for (x = 0, baseY = sizeX*z; x < sizeX; x+=yPencilsPerSlab, baseY++) {
	    for(z = 0; z < xPencilsPerSlab; z++){
		for(y = 0; y < yPencilsPerSlab; y++)
		    sendData[y] = line[y+baseY];
		//	memcpy(sendData, (line+x+sizeX*z), sizeof(complex)*yPencilsPerSlab);
	    }
	    
#ifdef VERBOSE
	    CkPrintf(" [%d %d] sending to YLINES [%d %d] \n",  thisIndex.x, thisIndex.y, thisIndex.y, x);
#endif
	    yProxy(zpos, x).doSecondFFT(thisIndex.x, sendData, sendDataSize, fftid, direction);
	}
	delete [] sendData;
    }
    else 
    {
	int  sendDataSize = yPencilsPerSlab;
	complex *sendData = new complex[sendDataSize];
	int ypos = thisIndex.y;
	//for(z = 0; z < xPencilsPerSlab; z++){
	    for (x = 0, baseY = sizeX*z; x < sizeX; x++, baseY++) {
		for(y = 0; y < yPencilsPerSlab; y++)
		    sendData[y] = line[y+baseY];
	    
#ifdef VERBOSE
		CkPrintf(" [%d %d] sending to YLINES [%d %d] \n",  thisIndex.x, thisIndex.y, thisIndex.y, x);
#endif
		yProxy(thisIndex.y+z, x).doSecondFFT(thisIndex.x, sendData, sendDataSize, fftid, direction);
	    }
	    //}
	delete [] sendData;
    }

#ifdef COMLIB
    mgrProxy.ckLocalBranch()->endIteration();
#endif
    
}

void
NormalLineArray::doSecondFFT(int ypos, complex *val, int datasize, int fftid, int direction)
{
    LineFFTinfo &fftinfo = (infoVec[fftid]->info);        
    complex *line = fftinfo.dataPtr;
    int sizeY = fftinfo.sizeY;
    int xPencilsPerSlab = fftinfo.xPencilsPerSlab;
    int yPencilsPerSlab = fftinfo.yPencilsPerSlab;
    int zPencilsPerSlab = fftinfo.zPencilsPerSlab;
    int	expectSize = 0;
    int x,z,baseZ;
    expectSize = yPencilsPerSlab*xPencilsPerSlab;
    CkAssert(datasize == expectSize);
    for(x=0; x<yPencilsPerSlab; x++)
	line[x*sizeY+ypos] = val[x];

    infoVec[fftid]->count++;
    if (infoVec[fftid]->count == sizeY/xPencilsPerSlab) {
	infoVec[fftid]->count = 0;
	int y;
	if(direction)
	    fftw(fwdplan, yPencilsPerSlab, (fftw_complex*)line, 1, sizeY, NULL, 0, 0);
	else
	    fftw(bwdplan, yPencilsPerSlab, (fftw_complex*)line, 1, sizeY, NULL, 0, 0);
	
#ifdef VERBOSE
	CkPrintf("Second FFT done at [%d %d]\n", thisIndex.x, thisIndex.y);
#endif
		// thisIndex.y is x-coord
#ifdef COMLIB
	CkAssert(id != -1);
	ckout << "ylines " << id << " " << CkMyPe() << endl;
	mgrProxy.ckLocalBranch()->beginIteration(id);
#endif
	if(direction){
	    int  sendDataSize = zPencilsPerSlab;
	    complex *sendData = new complex[sendDataSize];
	    int xpos = thisIndex.y;
	    for (y = 0, baseZ = x*sizeY; y < sizeY; y++, baseZ++) {
		//for (x = 0; x < yPencilsPerSlab; x++){
                    for(z = 0; z < zPencilsPerSlab; z++)
                        sendData[z] = line[z+baseZ];
		    //}
#ifdef VERBOSE
		CkPrintf(" [%d %d] sending to ZLINES [%d %d] \n",  thisIndex.x, thisIndex.y, thisIndex.y, y);
#endif	
		(zProxy)(xpos, y).doThirdFFT(thisIndex.x, 0, sendData, sendDataSize, fftid, direction);
		
	    }
	    delete [] sendData;
	}
	else {
	    int  sendDataSize = xPencilsPerSlab;
	    complex *sendData = new complex[sendDataSize];
	    for (x = 0; x < yPencilsPerSlab; x++){
		for (y = 0; y < sizeY; y+=xPencilsPerSlab) {
#ifdef VERBOSE
		CkPrintf(" [%d %d] sending to XLINES [%d %d] \n",  thisIndex.x, thisIndex.y, y, thisIndex.y);
#endif
		(xProxy)(y, thisIndex.x).doThirdFFT(thisIndex.y, y, &line[y], sendDataSize, fftid, direction);
		}
	    }
	    delete [] sendData;
	}
#ifdef COMLIB
	mgrProxy.ckLocalBranch()->endIteration();
#endif
    }
}

void
NormalLineArray::doThirdFFT(int zpos, int ypos, complex *val, int datasize, int fftid, int direction)
{
    LineFFTinfo &fftinfo = (infoVec[fftid]->info);        
    complex *line = fftinfo.dataPtr;
    int sizeX = fftinfo.sizeX;
    int sizeZ = fftinfo.sizeZ;
    int xPencilsPerSlab = fftinfo.xPencilsPerSlab;
    int zPencilsPerSlab = fftinfo.zPencilsPerSlab;
    int expectSize=0, offset=0, i; 
    if(direction){
	expectSize = zPencilsPerSlab;
	for(int y=0; y<zPencilsPerSlab; y++)
	    line[zpos+y*sizeZ] = val[y];
	offset = zpos;
    }
    else{
	expectSize = xPencilsPerSlab;
	for(i=0,offset=ypos; i<expectSize; i++,offset+=sizeX)
	    line[offset] = val[i]; //
    }
    CkAssert(datasize == expectSize);


    infoVec[fftid]->count += expectSize;

    if (infoVec[fftid]->count == sizeX* expectSize) {
	infoVec[fftid]->count = 0;

	memset(line, 1, sizeof(complex)*fftinfo.sizeX*fftinfo.zPencilsPerSlab);

	if(direction)
	    fftw(fwdplan, zPencilsPerSlab, (fftw_complex*)line, 1, sizeX, NULL, 0, 0);
	else
	    fftw(bwdplan, xPencilsPerSlab, (fftw_complex*)line, 1, sizeX, NULL, 0, 0); // sPencilsPerSlab many 1-D fft's 
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

NormalLineArray::NormalLineArray (LineFFTinfo &info, CProxy_NormalLineArray _xProxy, CProxy_NormalLineArray _yProxy, CProxy_NormalLineArray _zProxy, bool _useCommlib, ComlibInstanceHandle &inst) {
    setup(info, _xProxy, _yProxy, _zProxy, _useCommlib, inst);
}

void
NormalLineArray::setup (LineFFTinfo &info, CProxy_NormalLineArray _xProxy, CProxy_NormalLineArray _yProxy, CProxy_NormalLineArray _zProxy, bool _useCommlib, ComlibInstanceHandle &inst) {
	xProxy = _xProxy;
	yProxy = _yProxy;
	zProxy = _zProxy;
	
	PencilArrayInfo *pencilinfo = new PencilArrayInfo();
	pencilinfo->info = info;
	pencilinfo->count = 0;
	pencilinfo->fftcommInstance = inst;
	infoVec.insert(infoVec.size(), pencilinfo);
	
	line = NULL;
	fwdplan = fftw_create_plan(info.sizeX, FFTW_FORWARD, FFTW_IN_PLACE);
	bwdplan = fftw_create_plan(info.sizeY, FFTW_BACKWARD, FFTW_IN_PLACE);
	id = -1;
	fftuseCommlib = _useCommlib;
}
