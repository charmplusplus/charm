
#include "patch.h"

///
///  \brief This entry function initializes the patch grid array. The
///  grid array tells the patch which messages to send to which PME
///  array elements. First the X,Y,Z bounds for in the charge grid is
///  computed and then all the intersecting X Pencils are isolated.
///  This function currently assumes each patch evenly divides the
///  charge grid.
///
void Patch::initialize (PatchInfo &info) {  
  _info = info;
  
  int start_y, start_z, size_y, size_z;
  
  start_y = (_info.fftinfo.sizeY * _info.my_y) / _info.ny;
  size_y  = (_info.fftinfo.sizeY + _info.ny - 1) /  _info.ny;
  
  start_z = (_info.fftinfo.sizeZ * _info.my_z) / _info.nz;
  size_z  = (_info.fftinfo.sizeZ + _info.nz - 1) / _info.nz;

  ///
  /// \brief Since we have an interpolation grid with periodic boundry
  /// conditions, we have to wrap some charges around
  ///  
  size_y += 2*INTERPOLATION_SIZE;
  size_z += 2*INTERPOLATION_SIZE;   
  start_y -= INTERPOLATION_SIZE;
  start_z -= INTERPOLATION_SIZE;
  
  //number of x pencils along y and z dimensions
  int nz_pblocks, ny_pblocks;
  nz_pblocks = (_info.fftinfo.sizeY / _info.fftinfo.grainY);
  ny_pblocks = (_info.fftinfo.sizeZ / _info.fftinfo.grainZ); 
  
  //Find all pencils that the patch will send messages to
  int *pencilmap = new int [nz_pblocks * ny_pblocks];
  memset (pencilmap, 0, (nz_pblocks * ny_pblocks) * sizeof(int));

  int ngrid_msgs = 0;
  int y = 0, z = 0;

  int ywrap = ((INTERPOLATION_SIZE + _info.fftinfo.grainY -1) /
	       _info.fftinfo.grainY);
  int zwrap = ((INTERPOLATION_SIZE + _info.fftinfo.grainZ -1) / 
	       _info.fftinfo.grainZ);

#if __TEST_PME_VERBOSE__
  CkPrintf ("%d: In initialize (starty, startz, sizey, sizez)=(%d, %d, %d, %d)\n", 
	    CkMyPe(), start_y, start_z, size_y, size_z);
  CkPrintf ("%d: In initialize (ywrap, zwrap) = (%d, %d)\n", CkMyPe(), 
	    ywrap, zwrap);
  CkPrintf ("%d: In initialize (ny_pblocks, nz_pblocks) = (%d, %d)\n", 
	    CkMyPe(), ny_pblocks, nz_pblocks);
#endif

  for (z = -zwrap; z < nz_pblocks + zwrap; z++) {
    if ( (z * _info.fftinfo.grainZ >= start_z) &&
	 (z * _info.fftinfo.grainZ <= start_z + size_z))
      for (y = -ywrap; y < ny_pblocks + ywrap; y++) 
	if ( (y * _info.fftinfo.grainY >= start_y) &&
	     (y * _info.fftinfo.grainY <= start_y + size_y)) 
	  {
	    int rel_y = (y + ny_pblocks) % ny_pblocks;
	    int rel_z = (z + nz_pblocks) % nz_pblocks;
	    
	    int index = rel_z * ny_pblocks + rel_y;	  
	    if (!pencilmap[index]) {
	      pencilmap[index] = 1;
	      ngrid_msgs ++;
	    }
	  }    
  }
  _nGridMsgs = ngrid_msgs;
  
  contribute (sizeof(int) * nz_pblocks * ny_pblocks, 
	      pencilmap, CkReduction::sum_int, _info.cb_start);
  
  _gridList = new LineFFTGrid [ngrid_msgs];
  
  CkCallback cb(CkIndex_Patch::receiveGrid(NULL), thisProxy[CkMyPe()]);
  ngrid_msgs = 0;
  for (z = -zwrap; z < nz_pblocks + zwrap; z++) {
    if ( (z * _info.fftinfo.grainZ >= start_z) &&
	 (z * _info.fftinfo.grainZ <= start_z + size_z))
      for (y = -ywrap; y < ny_pblocks + ywrap; y++) 
	if ( (y * _info.fftinfo.grainY >= start_y) &&
	     (y * _info.fftinfo.grainY <= start_y + size_y)) 
	  {
	    int rel_y = (y + ny_pblocks) % ny_pblocks;
	    int rel_z = (z + nz_pblocks) % nz_pblocks;
	    
	    int index = rel_z * ny_pblocks + rel_y;	  
	    if (pencilmap[index] == 1) {
	      pencilmap [index] --;
	      _gridList [ngrid_msgs].y0 = rel_y * _info.fftinfo.grainY;
	      _gridList [ngrid_msgs].z0 = rel_z * _info.fftinfo.grainZ;
	      _gridList [ngrid_msgs].ysize = _info.fftinfo.grainY;
	      _gridList [ngrid_msgs].zsize = _info.fftinfo.grainZ;
	      
	      _gridList [ngrid_msgs].idx_x = rel_y;
	      _gridList [ngrid_msgs].idx_y = rel_z;	    
	      _gridList [ngrid_msgs].cb_done = cb;
	      
	      ngrid_msgs ++;
	    }
	  }    
  }

  CkAssert (ngrid_msgs == _nGridMsgs);

  int start_x = 0, size_x = 0;

  start_x = (_info.fftinfo.sizeX * _info.my_x) / _info.nx;
  size_x  = (_info.fftinfo.sizeX + _info.nx - 1) /  _info.nx;

  size_x += 2*INTERPOLATION_SIZE;
  start_x -= INTERPOLATION_SIZE;

  for (int count = 0; count < ngrid_msgs; count ++) {
    _gridList [count].x0 = start_x;
    _gridList [count].xsize = size_x;    
  }

  delete [] pencilmap;  
}


///
///  \brief Start the time step by sending grid messages to the Pencil
///  FFT Chare Array elements. 
///
void Patch :: startTimeStep () {

  for (int count = 0; count < _nGridMsgs; count ++) {
    int size = _gridList[count].xsize * 
      _gridList[count].ysize * _gridList[count].zsize;
    
    LineFFTGridMsg *msg = new (size, sizeof(int) * 8) LineFFTGridMsg();
    msg->grid = _gridList [count];
    
    for (int idx = 0; idx < size; idx ++)
      msg->data[idx] = 2.0;
    
    _info.fftinfo.xProxy (_gridList[count].idx_x, _gridList[count].idx_y).
      receiveGridMessage (msg);
  }
}

///
/// \brief Receive the fft'ed messages from Pencil FFT Chare array
/// elements
///
void Patch :: receiveGrid (LineFFTGridMsg *msg) {

  delete msg;
  _nReceived ++;
  
  if (_nReceived == _nGridMsgs) {
    _nReceived = 0;

#if __TEST_PME_VERBOSE__
    CkPrintf ("%d: Finished Iteration %d, max iterations = %d, ngridmsgs=%d\n", 
	      CkMyPe(), _iteration, _info.niterations, _nGridMsgs);
#endif

    _iteration ++;    
    if (_iteration >= _info.niterations) {
      int x = 0;
      contribute (sizeof(int), &x, CkReduction::sum_int, _info.cb_done);
      return;
    }
    
    startTimeStep ();
  }
};
