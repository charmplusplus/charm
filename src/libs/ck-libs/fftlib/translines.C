#include "fftlib.h"

//#define HEAVYVERBOSE 1
//#define VERBOSE 1
//#define _PRIOMSG_

void
NormalLineArray::doFirstFFT(int fftid, int direction)
{
    LineFFTinfo &fftinfo = (infoVec[fftid]->info);
    int ptype = fftinfo.ptype;
    int pblock = fftinfo.pblock;
    complex *line = fftinfo.dataPtr;
    int sizeX = fftinfo.sizeX;
    int sizeZ = fftinfo.sizeZ;
    int *xsquare = fftinfo.xsquare;
    int *ysquare = fftinfo.ysquare;
    int *zsquare = fftinfo.zsquare;

#ifdef HEAVYVERBOSE 
    {
	char fname[80];
	if(direction)
	snprintf(fname,80,"xline_%d.y%d.z%d.out", fftid,thisIndex.x, thisIndex.y);
	else
	snprintf(fname,80,"zline_%d.x%d.y%d.out", fftid,thisIndex.x, thisIndex.y);
      FILE *fp=fopen(fname,"w");
	for(int x = 0; x < sizeX*xsquare[0]*xsquare[1]; x++)
	    fprintf(fp, "%d  %g %g\n", x, line[x].re, line[x].im);
	fclose(fp);
    }
#endif

    if(direction && ptype==PencilType::XLINE)
	fftw(fwdplan, xsquare[0]*xsquare[1], (fftw_complex*)line, 1, sizeX, NULL, 0, 0); // xPencilsPerSlab many 1-D fft's 
    else if(!direction && ptype==PencilType::ZLINE)
	fftw(bwdplan, zsquare[0]*zsquare[1], (fftw_complex*)line, 1, sizeZ, NULL, 0, 0);
    else
	CkAbort("Can't do this FFT\n");

    int x, y, z=0;
#ifdef VERBOSE
    CkPrintf("First FFT done at [%d %d] [%d %d]\n", thisIndex.x, thisIndex.y,sizeX,sizeZ);
#endif
    int baseX, ix, iy, iz;
    if(true) {//else if(pblock == PencilBlock::SQUAREBLOCK){
	if(direction)
	{
	    int sendSquarethick = ysquare[1] <= xsquare[1] ? ysquare[1]:xsquare[1];
	    int sendDataSize = ysquare[0]*xsquare[0] * sendSquarethick;
	    int zpos = thisIndex.y;
	    int index=0;
	    complex *sendData = NULL;

	    for(z = 0; z < xsquare[1]; z+=sendSquarethick){
		for(x = 0; x < sizeX; x+=ysquare[0]) {
		    SendFFTMsg *msg = new(sendDataSize, sizeof(int)*8) SendFFTMsg;
		    sendData = msg->data;
		    msg->ypos = thisIndex.x; 
		    msg->size = sendDataSize;
		    msg->id = fftid;
		    msg->direction = direction;
		    msg->data = sendData;
		    CkSetQueueing(msg, CK_QUEUEING_IFIFO);
#ifdef _PRIOMSG_
		    int prioNum =  (zpos+z) + x*sizeX;
		    *(int*)CkPriorityPtr(msg) = prioNum; 
#endif	
		    index = 0;
		    for(iz = z; iz < z+sendSquarethick; iz++)
			for(ix = x; ix < x+ysquare[0]; ix++)
			    for(y = 0; y < xsquare[0]; y++)
				sendData[index++] = line[iz*sizeX*xsquare[0]+y*sizeX+ix];
		
#ifdef VERBOSE	
		    CkPrintf(" [%d %d] sending to YLINES [	%d %d] \n",  thisIndex.x, thisIndex.y, thisIndex.y, x);
#endif	
		    yProxy(zpos+z, x).doSecondFFT(msg);
		}
	    //memset(sendData, 0, sizeof(complex)*yPencilsPerSlab*xPencilsPerSlab);
	    }
	}
	else
	{
	    int sendSquarewidth = ysquare[0]<=zsquare[0] ? ysquare[0]:zsquare[0];
	    int sendDataSize = ysquare[1] * sendSquarewidth * zsquare[1];
	    int xpos = thisIndex.x;
	    int ypos = thisIndex.y;
	    int index=0;
	    complex *sendData = NULL;

	    for(x = 0; x < zsquare[0]; x+=sendSquarewidth)
		for(z = 0; z < sizeZ; z+=ysquare[1]){
		    SendFFTMsg *msg = new(sendDataSize, sizeof(int)*8) SendFFTMsg;
		    sendData = msg->data;
		    msg->ypos = thisIndex.y; 
		    msg->size = sendDataSize;
		    msg->id = fftid;
		    msg->direction = direction;
		    msg->data = sendData;
		    CkSetQueueing(msg, CK_QUEUEING_IFIFO);
#ifdef _PRIOMSG_
		    int prioNum =  (z) + (x+xpos)*sizeX;
		    *(int*)CkPriorityPtr(msg) = prioNum; 
#endif	
		    index = 0;
		    for(iz = z; iz < z+ysquare[1]; iz++)
			for (ix = x; ix < x+sendSquarewidth; ix++) 
			    for(iy = 0; iy < zsquare[1]; iy++)
				sendData[index++] = line[iz+ix*sizeZ+iy*sizeZ*zsquare[0]];
		
#ifdef VERBOSE	
		    CkPrintf(" [%d %d] sending	 to YLINES [%d %d] \n",  thisIndex.x, thisIndex.y, z, thisIndex.x);
#endif
		    yProxy(z, xpos+x).doSecondFFT(msg);
		}
	}
    }
}

void
NormalLineArray::doSecondFFT(int ypos, complex *val, int datasize, int fftid, int direction)
{
    LineFFTinfo &fftinfo = (infoVec[fftid]->info);        
    int ptype = fftinfo.ptype;
    complex *line = fftinfo.dataPtr;
    int sizeY = fftinfo.sizeY;
    int *xsquare = fftinfo.xsquare;
    int *ysquare = fftinfo.ysquare;
    int *zsquare = fftinfo.zsquare;
    int expectSize = 0, expectMsg = 0;
    int x,y,z,baseY;
    int ix,iy,iz;
    if(direction){
	int sendSquarethick = ysquare[1]<=xsquare[1] ? ysquare[1]:xsquare[1];
	expectSize = ysquare[0]*xsquare[0] * sendSquarethick;
	expectMsg = sizeY/xsquare[0] * (ysquare[1]/sendSquarethick);
	CkAssert(datasize == expectSize);
	int idx = 0;
	for(z=0; z<sendSquarethick; z++)
	    for(x=0; x<ysquare[0]; x++)
		for(y=0; y<xsquare[0]; y++)
		    line[z*sizeY*ysquare[0]+x*sizeY+ypos+y] = val[idx++];
    }
    else {
	int sendSquarewidth = ysquare[0]<=zsquare[0] ? ysquare[0]:zsquare[0];
	expectSize = ysquare[1] * sendSquarewidth * zsquare[1];
	expectMsg = sizeY/zsquare[1] * (ysquare[0]/sendSquarewidth);
	CkAssert(datasize == expectSize);
	int idx=0;
	for(z=0; z<ysquare[1]; z++)
	    for(x=0; x<sendSquarewidth; x++)
		for(y=0; y<zsquare[1]; y++)
		    line[z*sizeY*ysquare[0]+x*sizeY+ypos+y] = val[idx++];
    }
    infoVec[fftid]->count++;
    if (infoVec[fftid]->count == expectMsg) {
	infoVec[fftid]->count = 0;
	int y;


    if(direction && ptype==PencilType::YLINE)
	    fftw(fwdplan, ysquare[0]*ysquare[1], (fftw_complex*)line, 1, sizeY, NULL, 0, 0);
    else if(!direction && ptype==PencilType::YLINE)
	    fftw(bwdplan, ysquare[0]*ysquare[1], (fftw_complex*)line, 1, sizeY, NULL, 0, 0);
    else
	CkAbort("Can't do this FFT\n");

#ifdef HEAVYVERBOSE
    {
	char fname[80];
	snprintf(fname,80,"yline_%d.x%d.z%d.out", fftid, thisIndex.y, thisIndex.x);
      FILE *fp=fopen(fname,"w");
	for(int x = 0; x < sizeY*ysquare[0]*ysquare[1]; x++)
	    fprintf(fp, "%d  %g %g\n", x, line[x].re, line[x].im);
	fclose(fp);
    }
#endif

#ifdef VERBOSE
	CkPrintf("Second FFT done at [%d %d]\n", thisIndex.x, thisIndex.y);
#endif
		// thisIndex.y is x-coord
	if(direction){
	    int sendSquarewidth = ysquare[0]<=zsquare[0] ? ysquare[0]:zsquare[0];
	    int sendDataSize = sendSquarewidth * ysquare[1] * zsquare[1];
	    int xpos = thisIndex.y;
	    int index=0;
	    complex *sendData = NULL;

	    for(x = 0; x < ysquare[0]; x+=sendSquarewidth)
		for(y = 0; y < sizeY; y+=zsquare[1]) {
		    SendFFTMsg *msg = new(sendDataSize, sizeof(int)*8) SendFFTMsg;
		    sendData = msg->data;
		    msg->zpos = thisIndex.x; 
		    msg->ypos = 0;
		    msg->size = sendDataSize;
		    msg->id = fftid;
		    msg->direction = direction;
		    msg->data = sendData;
		    CkSetQueueing(msg, CK_QUEUEING_IFIFO);
#ifdef _PRIOMSG_
		    int prioNum =  (xpos+x) + y*sizeY;
		    *(int*)CkPriorityPtr(msg) = prioNum; 
#endif	
		    index = 0;
		    for(iy = y; iy < y+zsquare[1]; iy++)
			for(ix = x; ix < x+sendSquarewidth; ix++)
			    for(iz = 0; iz < ysquare[1]; iz++)
				sendData[index++] = line[iz*sizeY*ysquare[0]+ix*sizeY+iy];
		
#ifdef VERBOSE
		    CkPrintf(" [%d %d] sending to ZLINES [	%d %d] \n",  thisIndex.x, thisIndex.y, thisIndex.y, y);
#endif	
		    (zProxy)(x+xpos, y).doThirdFFT(msg);
		}
	}
	else {
	    int sendSquarethick = ysquare[1]<=xsquare[1] ? ysquare[1]:xsquare[1];
	    int sendDataSize = ysquare[0]*xsquare[0] * sendSquarethick;
	    int zpos = thisIndex.x;
	    int index, ix, iy, iz;
	    complex *sendData = NULL;

	    for(z = 0; z < ysquare[1]; z+=sendSquarethick)
		for (y = 0; y < sizeY; y+=xsquare[0]) {
		    SendFFTMsg *msg = new(sendDataSize, sizeof(int)*8) SendFFTMsg;
		    sendData = msg->data;
		    msg->zpos = 0;
		    msg->ypos = thisIndex.y;
		    msg->size = sendDataSize;
		    msg->id = fftid;
		    msg->direction = direction;
		    msg->data = sendData;
		    CkSetQueueing(msg, CK_QUEUEING_IFIFO);
#ifdef _PRIOMSG_		    
		    int prioNum =  (y)*sizeY + (zpos+z);
		    *(int*)CkPriorityPtr(msg) = prioNum; 
#endif	
		    index = 0;
		    for(iz = z; iz < z+sendSquarethick; iz++)
			for(iy = y; iy < y+xsquare[0]; iy++)
			    for(x = 0; x < ysquare[0]; x++)
				sendData[index++] = line[iz*sizeY*ysquare[0]+x*sizeY+iy];

#ifdef VERBOSE
		    CkPrintf(" [%d %d] sending to XLINES [%d %d]	 \n",  thisIndex.x, thisIndex.y, y, zpos+z);
#endif
		    (xProxy)(y, zpos+z).doThirdFFT(msg);
		}
	}	
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
    int *xsquare = fftinfo.xsquare;
    int *ysquare = fftinfo.ysquare;
    int *zsquare = fftinfo.zsquare;
    int expectSize=0, expectMsg=0, offset=0, i; 

    int x,y,z,idx;
    if(direction){
	int sendSquarewidth = ysquare[0]<=zsquare[0] ? ysquare[0]:zsquare[0];
	expectSize = sendSquarewidth * ysquare[1] * zsquare[1];
	expectMsg = sizeZ/ysquare[1] * (zsquare[0]/sendSquarewidth);
	CkAssert(datasize == expectSize);
	idx=0;
	for(y=0; y<zsquare[1]; y++)
	    for(x=0; x<sendSquarewidth; x++)
		for(z=0; z<ysquare[1]; z++)
		    line[z+zpos+(x+xpos)*sizeZ+y*sizeZ*zsquare[0]] = val[idx++];
    }
    else{
	int sendSquarethick = ysquare[1]<=xsquare[1] ? ysquare[1]:xsquare[1];
	expectSize = ysquare[0]*xsquare[0] * sendSquarethick;
	expectMsg = sizeX/ysquare[0] * (xsquare[1]/sendSquarethick);
	CkAssert(datasize == expectSize);
	int idx=0;
	for(z=0; z<sendSquarethick; z++)
	    for(y=0; y<xsquare[0]; y++)
		for(x=0; x<ysquare[0]; x++)
		    line[(z+zpos)*sizeX*xsquare[0]+y*sizeX+xpos+x] = val[idx++];
    }

    infoVec[fftid]->count ++;

    if (infoVec[fftid]->count == expectMsg) {
	infoVec[fftid]->count = 0;


#ifdef HEAVYVERBOSE
	{
	char fname[80];
	if(direction)
	snprintf(fname,80,"zline_%d.x%d.y%d.out", fftid, thisIndex.x, thisIndex.y);
	else
	snprintf(fname,80,"xline_%d.y%d.z%d.out", fftid, thisIndex.x, thisIndex.y);
      FILE *fp=fopen(fname,"w");
	for(int x = 0; x < sizeX*xsquare[0]*xsquare[1]; x++)
	    fprintf(fp, "%g %g\n", line[x].re, line[x].im);
	fclose(fp);
	}
#endif


	if(direction && ptype==PencilType::ZLINE)
	    fftw(fwdplan, zsquare[0]*zsquare[1], (fftw_complex*)line, 1, sizeX, NULL, 0, 0);
	else if(!direction && ptype==PencilType::XLINE)
	    fftw(bwdplan, xsquare[0]*xsquare[1], (fftw_complex*)line, 1, sizeX, NULL, 0, 0); // sPencilsPerSlab many 1-D fft's 
	else
	    CkAbort("Can't do this FFT\n");
#ifdef VERBOSE
	CkPrintf("Third FFT done at [%d %d]\n", thisIndex.x, thisIndex.y);
#endif
	doneFFT(fftid, direction);
//	contribute(sizeof(int), &count, CkReduction::sum_int);
    }
}

void 
NormalLineArray::doSecondFFT(SendFFTMsg *msg){
    doSecondFFT(msg->ypos, msg->data, msg->size, msg->id, msg->direction);
    delete msg;
}

void 
NormalLineArray::doThirdFFT(SendFFTMsg *msg){
    doThirdFFT(msg->zpos, msg->ypos, msg->data, msg->size, msg->id, msg->direction);
    delete msg;
}

void 
NormalLineArray::doneFFT(int id, int direction){
#ifdef VERBOSE
    CkPrintf("FFT finished \n");
#endif
}

NormalLineArray::NormalLineArray (LineFFTinfo &info, CProxy_NormalLineArray _xProxy, CProxy_NormalLineArray _yProxy, CProxy_NormalLineArray _zProxy) {
#ifdef VERBOSE
    CkPrintf("inserted line %d index[%d %d]\n", info.ptype, thisIndex.x, thisIndex.y);
#endif
    setup(info, _xProxy, _yProxy, _zProxy);
}

void
NormalLineArray::setup (LineFFTinfo &info, CProxy_NormalLineArray _xProxy, CProxy_NormalLineArray _yProxy, CProxy_NormalLineArray _zProxy) {
	xProxy = _xProxy;
	yProxy = _yProxy;
	zProxy = _zProxy;
	
	PencilArrayInfo *pencilinfo = new PencilArrayInfo();
	pencilinfo->info = info;
	pencilinfo->count = 0;
	infoVec.insert(infoVec.size(), pencilinfo);
	
	line = NULL;
	fwdplan = fftw_create_plan(info.sizeX, FFTW_FORWARD, FFTW_IN_PLACE);
	bwdplan = fftw_create_plan(info.sizeY, FFTW_BACKWARD, FFTW_IN_PLACE);
	id = -1;
}
