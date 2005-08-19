#include "fftlib.h"

//#define VERBOSE 1

void
NormalLineArray::doFirstFFT(int fftid, int direction)
{
    LineFFTinfo &fftinfo = (infoVec[fftid]->info);
    int ptype = fftinfo.ptype;
    complex *line = fftinfo.dataPtr;
    int sizeX = fftinfo.sizeX;
    int sizeZ = fftinfo.sizeZ;
    int xPencilsPerSlab = fftinfo.xPencilsPerSlab;
    int yPencilsPerSlab = fftinfo.yPencilsPerSlab;
    int zPencilsPerSlab = fftinfo.zPencilsPerSlab;

    if(direction && ptype==PencilType::XLINE)
	fftw(fwdplan, xPencilsPerSlab, (fftw_complex*)line, 1, sizeX, NULL, 0, 0); // xPencilsPerSlab many 1-D fft's 
    else if(!direction && ptype==PencilType::ZLINE)
	fftw(bwdplan, zPencilsPerSlab, (fftw_complex*)line, 1, sizeZ, NULL, 0, 0);
    else
	CkAbort("Can't do this FFT\n");
#if 0
    {
	char fname[80];
	snprintf(fname,80,"xline.y%d.z%d.out", thisIndex.x, thisIndex.y);
      FILE *fp=fopen(fname,"w");
	for(int x = 0; x < sizeX*xPencilsPerSlab; x++)
	    fprintf(fp, "%d  %g %g\n", x, line[x].re, line[x].im);
	fclose(fp);
    }
#endif

    int x, y, z=0;
#ifdef VERBOSE
    CkPrintf("First FFT done at [%d %d] [%d %d]\n", thisIndex.x, thisIndex.y,sizeX,sizeZ);
#endif
#ifdef COMLIB
    CkAssert(id != -1);
    ckout << "xlines " << id << endl;
    mgrProxy.ckLocalBranch()->beginIteration(id);
#endif
    int baseX, ix;
    if(direction)
    {
	int  sendDataSize = yPencilsPerSlab*xPencilsPerSlab;
	complex *sendData = new complex[sendDataSize];
	int zpos = thisIndex.y;
	int idx=0;
	for (x = 0; x < sizeX; x+=yPencilsPerSlab) {
	    for(idx=0, ix = x; ix < x+yPencilsPerSlab; ix++)
		for(y = 0; y < xPencilsPerSlab; y++)
		    sendData[idx++] = line[y*sizeX+ix];
	    
#ifdef VERBOSE
	    CkPrintf(" [%d %d] sending to YLINES [%d %d] \n",  thisIndex.x, thisIndex.y, thisIndex.y, x);
#endif
	    yProxy(zpos, x).doSecondFFT(thisIndex.x, sendData, sendDataSize, fftid, direction);
	    memset(sendData, 0, sizeof(complex)*yPencilsPerSlab*xPencilsPerSlab);
	}
	delete [] sendData;
    }
    else 
    {
	int  sendDataSize = (yPencilsPerSlab<=zPencilsPerSlab) ? 
	    yPencilsPerSlab:zPencilsPerSlab;
	complex *sendData = new complex[sendDataSize];
	int ypos = thisIndex.y;
	for(z = 0; z < sizeZ; z++){
	    for (x = 0; x < sendDataSize; x++) 
		sendData[x] = line[z+x*sizeZ];

#ifdef VERBOSE
	    CkPrintf(" [%d %d] sending to YLINES [%d %d] \n",  thisIndex.x, thisIndex.y, z, thisIndex.x);
#endif
	    yProxy(z, thisIndex.x).doSecondFFT(ypos, sendData, sendDataSize, fftid, direction);
	}
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
    int ptype = fftinfo.ptype;
    complex *line = fftinfo.dataPtr;
    int sizeY = fftinfo.sizeY;
    int xPencilsPerSlab = fftinfo.xPencilsPerSlab;
    int yPencilsPerSlab = fftinfo.yPencilsPerSlab;
    int zPencilsPerSlab = fftinfo.zPencilsPerSlab;
    int expectSize = 0, expectMsg = 0;
    int x,z,baseY;
    if(direction){
	expectSize = yPencilsPerSlab*xPencilsPerSlab;
	expectMsg = sizeY/xPencilsPerSlab;
	CkAssert(datasize == expectSize);
	int idx;
	for(idx=0, x=0; x<yPencilsPerSlab; x++)
	    for(int y=0; y<xPencilsPerSlab; y++)
		line[x*sizeY+ypos+y] = val[idx++];
    }
    else {
	expectSize = (yPencilsPerSlab<=zPencilsPerSlab) ? 
	    yPencilsPerSlab:zPencilsPerSlab; 
	expectMsg = sizeY;
	CkAssert(datasize == expectSize);
	int idx;
	for(idx=0, x=0; x<datasize; x++)
	    line[x*sizeY+ypos] = val[idx++];
    }
    infoVec[fftid]->count++;
    if (infoVec[fftid]->count == expectMsg) {
	infoVec[fftid]->count = 0;
	int y;

#if 0
    {
	char fname[80];
	snprintf(fname,80,"yline.x%d.z%d.out", thisIndex.y, thisIndex.x);
      FILE *fp=fopen(fname,"w");
	for(int x = 0; x < sizeY*yPencilsPerSlab; x++)
	    fprintf(fp, "%d  %g %g\n", x, line[x].re, line[x].im);
	fclose(fp);
    }
#endif

    if(direction && ptype==PencilType::YLINE)
	    fftw(fwdplan, yPencilsPerSlab, (fftw_complex*)line, 1, sizeY, NULL, 0, 0);
    else if(!direction && ptype==PencilType::YLINE)
	    fftw(bwdplan, yPencilsPerSlab, (fftw_complex*)line, 1, sizeY, NULL, 0, 0);
    else
	CkAbort("Can't do this FFT\n");

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
	    int  sendDataSize = (yPencilsPerSlab<=zPencilsPerSlab) ? 
		yPencilsPerSlab:zPencilsPerSlab;
	    complex *sendData = new complex[sendDataSize];
	    int xpos = thisIndex.y;
	    for (y = 0; y < sizeY; y++) {
		for(x = 0; x < sendDataSize; x++)
		    sendData[x] = line[x*sizeY+y];
		
#ifdef VERBOSE
		CkPrintf(" [%d %d] sending to ZLINES [%d %d] \n",  thisIndex.x, thisIndex.y, thisIndex.y, y);
#endif	
		(zProxy)(xpos, y).doThirdFFT(thisIndex.x, 0, sendData, sendDataSize, fftid, direction);
		
	    }
	    delete [] sendData;
	}
	else {
	    int  sendDataSize = xPencilsPerSlab*yPencilsPerSlab;
	    complex *sendData = new complex[sendDataSize];
	    int idx, iy;
	    for (y = 0; y < sizeY; y+=xPencilsPerSlab) {
		for(idx=0, iy = y; iy < y+xPencilsPerSlab; iy++)
		    for(x = 0; x < yPencilsPerSlab; x++)
			sendData[idx++] = line[x*sizeY+iy];

#ifdef VERBOSE
		CkPrintf(" [%d %d] sending to XLINES [%d %d] \n",  thisIndex.x, thisIndex.y, y, thisIndex.x);
#endif
		(xProxy)(y, thisIndex.x).doThirdFFT(0, thisIndex.y, sendData, sendDataSize, fftid, direction);
	    }
	    
	    delete [] sendData;
	}
#ifdef COMLIB
	mgrProxy.ckLocalBranch()->endIteration();
#endif
    }
}

void
NormalLineArray::doThirdFFT(int zpos, int xpos, complex *val, int datasize, int fftid, int direction)
{
    LineFFTinfo &fftinfo = (infoVec[fftid]->info);        
    int ptype = fftinfo.ptype;
    complex *line = fftinfo.dataPtr;
    int sizeX = fftinfo.sizeX;
    int sizeZ = fftinfo.sizeZ;
    int xPencilsPerSlab = fftinfo.xPencilsPerSlab;
    int yPencilsPerSlab = fftinfo.yPencilsPerSlab;
    int zPencilsPerSlab = fftinfo.zPencilsPerSlab;
    int expectSize=0, expectTotalSize=0, offset=0, i; 
    if(direction){
	expectSize = (yPencilsPerSlab<=zPencilsPerSlab) ? 
	    yPencilsPerSlab:zPencilsPerSlab;
	expectTotalSize = sizeZ*zPencilsPerSlab;
	CkAssert(datasize == expectSize);
	for(int y=0; y<datasize; y++)
	    line[zpos+y*sizeZ] = val[y];
    }
    else{
	expectSize = xPencilsPerSlab*yPencilsPerSlab;
	expectTotalSize = sizeX*xPencilsPerSlab;
	CkAssert(datasize == expectSize);
	int idx, y;
	for(idx=0, y=0; y<xPencilsPerSlab; y++)
	    for(int x=0; x<yPencilsPerSlab; x++)
		line[y*sizeX+xpos+x] = val[idx++];
    }

    infoVec[fftid]->count += expectSize;

    if (infoVec[fftid]->count == expectTotalSize) {
	infoVec[fftid]->count = 0;

//	memset(line, 1, sizeof(complex)*fftinfo.sizeX*fftinfo.zPencilsPerSlab);

#if 0
    {
	char fname[80];
	snprintf(fname,80,"zline.x%d.y%d.out", thisIndex.x, thisIndex.y);
      FILE *fp=fopen(fname,"w");
	for(int x = 0; x < sizeX*xPencilsPerSlab; x++)
	    fprintf(fp, "%d  %g %g\n", x, line[x].re, line[x].im);
	fclose(fp);
    }
#endif

	if(direction && ptype==PencilType::ZLINE)
	    fftw(fwdplan, zPencilsPerSlab, (fftw_complex*)line, 1, sizeX, NULL, 0, 0);
	else if(!direction && ptype==PencilType::XLINE)
	    fftw(bwdplan, xPencilsPerSlab, (fftw_complex*)line, 1, sizeX, NULL, 0, 0); // sPencilsPerSlab many 1-D fft's 
	else
	    CkAbort("Can't do this FFT\n");
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
