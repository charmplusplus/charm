
#include "pencilfft.h"

#define CONST_A  1001
#define CONST_B  17

///
/// \brief main constructor
///
LineFFTArray::LineFFTArray (LineFFTInfo &info, int phase): _info (info)
{
#if __LINEFFT_DEBUG__
  CkPrintf ("[%d, %d, %d] Pencil FFT array constructor \n", thisIndex.x, thisIndex.y, phase);
#endif

  _nElements = 0;
  _fftSize   = 0;
  _phase     = (LineFFTPhase) phase;

  switch (phase) {
  case PHASE_X: 
    _fftSize   = info.sizeX;
    _nElements = info.sizeX * info.grainY * info.grainZ;
    _nFFTMessages = info.sizeX / info.grainX;
    break;

  case PHASE_Y:
    _fftSize   = info.sizeY; 
    _nElements = info.sizeY * info.grainZ * info.grainX;
    _nFFTMessages = info.sizeY / info.grainY;
    break;

  case PHASE_Z: 
    _fftSize   = info.sizeZ; 
    _nElements = info.sizeZ * info.grainX * info.grainY;
    _nFFTMessages = info.sizeZ / info.grainZ;
    break;
  };    

  _line = new complex [_nElements];
  
  LineFFTAssert ((_nFFTMessages % CONST_B) != 0);

#if 0 //__LINEFFT_DEBUG__
  for (int count = 0; count < _nElements; count++)
    _line[count] = complex (1.0 * count, 0.0);
#endif

  _fwdplan = fftw_create_plan(_fftSize, FFTW_FORWARD, FFTW_IN_PLACE | 
			      FFTW_MEASURE | FFTW_USE_WISDOM);
  _bwdplan = fftw_create_plan(_fftSize, FFTW_BACKWARD, FFTW_IN_PLACE | 
			      FFTW_MEASURE | FFTW_USE_WISDOM);

  _curFFTMessages  = 0;
  _curGridMessages = 0;
  _nGridMessages   = 0;
}


///
/// \brief Receive an intermediate FFT message
///
void LineFFTArray::receiveFFTMessage (LineFFTMsg *msg) {

#if __LINEFFT_DEBUG__
  CkPrintf ("[%d, %d] Pencil FFT receive fft message \n", 
	    thisIndex.x, thisIndex.y);
#endif
  int index = 0, localindex=0;
  int start = 0;

#if __LINEFFT_DEBUG__
  LineFFTAssert (msg->phase == _phase);
#endif

  switch (_phase) {
  case PHASE_X:
    start = msg->chunk_id * _info.grainX;
    for (int z = 0; z < _info.grainZ; z++) 
      for (int y = 0; y < _info.grainY; y++) 
	for (int x = start; x < start + _info.grainX; x++) {
	  localindex = _info.sizeX * _info.grainY * z + _info.sizeX * y + x;
#if __LINEFFT_DEBUG__
	  LineFFTAssert (localindex < _nElements);
#endif
	  _line [localindex] = msg->data[index++];   
	}    
    break;
    
  case PHASE_Y:  
    start = msg->chunk_id * _info.grainY;
    for (int z = 0; z < _info.grainZ; z++) 
      for (int y = start; y < start+_info.grainY; y++) 
	for (int x = 0; x < _info.grainX; x++) {
	  localindex = _info.sizeY * _info.grainZ * x + _info.sizeY * z + y;
#if __LINEFFT_DEBUG__
	  LineFFTAssert (localindex < _nElements);
#endif
	  _line [localindex] = msg->data[index++];   
	}
    break;
    
  case PHASE_Z:
    start = msg->chunk_id * _info.grainZ;
    for (int z = start; z < start+_info.grainZ; z++) 
      for (int y = 0; y < _info.grainY; y++) 
	for (int x = 0; x < _info.grainX; x++) {
	  localindex = _info.sizeZ * _info.grainX * y + _info.sizeZ * x + z;
#if __LINEFFT_DEBUG__
	  LineFFTAssert (localindex < _nElements);
#endif
	  _line [localindex] = msg->data[index++];   
	}
    break;

  default:
    CkAbort ("Phase undefined");
    break;
  };
    
  _curFFTMessages ++;

  if (_curFFTMessages == _nFFTMessages) {
    _curFFTMessages = 0;
    startFFT (msg->direction);
  }

  CkFreeMsg (msg);
}

///
/// \brief Send the fft messages for the next few phases. Entry
/// constructs a LineFFTMsg by splitting the thick pencil into chunks
/// and then sends a chunk to a the neighbor intersecting on that
/// chunk. The message is always formatted in the XYZ order
/// 
void LineFFTArray::sendFFTMessages (int dir) {
  
  int size = _info.grainX * _info.grainY * _info.grainZ;
  CProxy_LineFFTArray proxy;
  int arridx_x = 0, arridx_y =0;

  for (int count = 0; count < _nFFTMessages; count ++) {
    int start = 0, index = 0, localindex =0;
    int chunk_id = (CONST_A*CkMyPe() + CONST_B * count) % _nFFTMessages;

    LineFFTMsg *msg = new (size, sizeof(int) * 8) LineFFTMsg;

    if (dir == FFTW_FORWARD) {
      arridx_x = thisIndex.y;
      arridx_y = chunk_id;
      msg->chunk_id  =  thisIndex.x;
    }
    else {
      arridx_x = chunk_id;
      arridx_y = thisIndex.x;
      msg->chunk_id  =  thisIndex.y;      
    }

    switch (_phase) {
    case PHASE_X:
      start = chunk_id * _info.grainX;
      for (int z = 0; z < _info.grainZ; z++) 
	for (int y = 0; y < _info.grainY; y++) 
	  for (int x = start; x < start+_info.grainX; x++) {
	    localindex = _info.sizeX * _info.grainY * z + _info.sizeX * y + x;
#if __LINEFFT_DEBUG__
	    LineFFTAssert (localindex < _nElements);
#endif
	    msg->data[index++] = _line [localindex];
	  }
  
      LineFFTAssert (dir == FFTW_FORWARD);
      proxy = _info.yProxy;
      msg->phase = PHASE_Y;
      
      break;
      
    case PHASE_Y:
      start = chunk_id * _info.grainY;
      for (int z = 0; z < _info.grainZ; z++) 
	for (int y = start; y < start+_info.grainY; y++) 
	  for (int x = 0; x < _info.grainX; x++) {
	    localindex = _info.sizeY * _info.grainZ * x + _info.sizeY * z + y;
#if __LINEFFT_DEBUG__
	    LineFFTAssert (localindex < _nElements);
#endif
	    msg->data[index++] = _line [localindex];
	  }

      if (dir == FFTW_FORWARD) {
	proxy = _info.zProxy;
	msg->phase = PHASE_Z;
      }
      else {	  
	proxy = _info.xProxy;
	msg->phase = PHASE_X;
      }

      break;
    
    case PHASE_Z:
      start = chunk_id * _info.grainZ;
      for (int z = start; z < start+_info.grainZ; z++) 
	for (int y = 0; y < _info.grainY; y++) 
	  for (int x = 0; x < _info.grainX; x++) {
	    localindex = _info.sizeZ * _info.grainX * y + _info.sizeZ * x + z;
#if __LINEFFT_DEBUG__
	    LineFFTAssert (localindex < _nElements);
#endif
	    msg->data[index++] = _line [localindex];
	  }

      LineFFTAssert (dir == FFTW_BACKWARD);
      proxy = _info.yProxy;
      msg->phase = PHASE_Y;

      break;
      
    default:
      CkAbort ("Phase undefined");
      break;
    };

    msg->direction =  dir;

    //Send data to the intersecting neighbor
    proxy(arridx_x, arridx_y).receiveFFTMessage (msg);
  }
  
}

///
/// \brief Start the 3D fft operation. This can be called externally
/// or intiated when all the grid/fft messages have been received.
///
void LineFFTArray::startFFT (int dir) {
  
#if __LINEFFT_DEBUG__
  CkPrintf ("[%d, %d, %d, %d] Pencil FFT start fft %d,%d\n", 
	    thisIndex.x, thisIndex.y, _phase, dir, _nElements, _fftSize);
#endif

  fftw_plan plan = NULL;
  
  if (dir == FFTW_FORWARD)
    plan = _fwdplan;
  else
    plan = _bwdplan;
  
  for (int count = 0; count < _nElements; count += _fftSize) {
    CmiNetworkProgress();
    fftw_one (plan, (fftw_complex *) (_line + count), NULL);
  }
  
  if (dir == FFTW_FORWARD && _phase == PHASE_Z) {
    if (_info.normalize) {
      fftw_real scale = 1.0 / (_info.sizeX * _info.sizeY * _info.sizeZ);
      for (int count = 0; count < _nElements; count ++)
	_line [count] *= scale;
    }
    
    //call_kspace_callback ();    
    startFFT (FFTW_BACKWARD);
    return;
  }

  if (dir == FFTW_BACKWARD && _phase == PHASE_X) {
#if __LINEFFT_DEBUG__
    for (int count = 0; count < _nElements; count++) {
      if (!((_line[count].re <= 1.001 * count)
	    && (_line[count].re >= 0.999 * count)))
	CkPrintf ("[%d, %d] val = (%5.3f, %5.3f) \n", thisIndex.x, thisIndex.y, 
		  _line[count].re, _line[count].im);
    }
#endif

    call_donecallback ();
    return;
  }
  
  sendFFTMessages (dir);
}


///
/// \brief Receive data points to be fft'ed. This will happen in the
/// first and last phases
///
void LineFFTArray::receiveGridMessage (LineFFTGridMsg   *msg) {  
  //copy grid data to the pencil lines
  LineFFTAssert (_phase == PHASE_X);

  int my_start_y = thisIndex.x * _info.grainY;
  int my_start_z = thisIndex.y * _info.grainZ;

  LineFFTGrid &grid = msg->grid;

  int index = 0;
  for (int z = grid.z0; z < grid.z0 + grid.zsize; z++){
    LineFFTAssert ((z >= my_start_z) && (z < my_start_z + _info.grainZ));
    for (int y = grid.y0; y < grid.y0 + grid.ysize; y++) {
      LineFFTAssert ((y >= my_start_y) && (y < my_start_y + _info.grainY));      
      for (int x = grid.x0; x < grid.x0 + grid.xsize; x++) {	    
	int rel_x = x;
	if (rel_x < 0)
	  rel_x = x + _info.sizeX;	
	if (rel_x >= _info.sizeX)
	  rel_x -= _info.sizeX;
	
	int local_index = (z - my_start_z) * _info.grainY * _info.sizeX +
	  (y - my_start_y) * _info.sizeX + rel_x;
	
	LineFFTAssert (msg->data [index] == 2.0);

	LineFFTAssert (local_index < _nElements);
	_line [local_index].re += msg->data [index++];
      }
    }
  }

  _gridList [_curGridMessages] = msg->grid;

  CkFreeMsg (msg);
  _curGridMessages ++;

  if (_curGridMessages == _nGridMessages) {
    _curGridMessages = 0;
    startFFT (FFTW_FORWARD);
  }
}


///
/// \brief Receive data points to be fft'ed. This will happen in the
/// first and last phases
///
void LineFFTArray::sendGridMessages () {  
  
  int my_start_y = thisIndex.x * _info.grainY;
  int my_start_z = thisIndex.y * _info.grainZ;

  for (int count = 0; count < _nGridMessages; count ++) {
    LineFFTGrid &grid = _gridList[count];

    LineFFTGridMsg *msg = 
      new (grid.xsize * grid.ysize * grid.zsize, sizeof(int) * 4) LineFFTGridMsg;
    
    int index = 0;
    for (int z = grid.z0; z < grid.z0 + grid.zsize; z++){
      LineFFTAssert ((z >= my_start_z) && (z < my_start_z + _info.grainZ));
      for (int y = grid.y0; y < grid.y0 + grid.ysize; y++) {
	LineFFTAssert ((y >= my_start_y) && (y < my_start_y + _info.grainY));      
	for (int x = grid.x0; x < grid.x0 + grid.xsize; x++) {	    
	  int rel_x = x;
	  if (rel_x < 0)
	    rel_x = x + _info.sizeX;	
	  if (rel_x >= _info.sizeX)
	    rel_x -= _info.sizeX;
	  
	  int local_index = (z - my_start_z) * _info.grainY * _info.sizeX +
	    (y - my_start_y) * _info.sizeX + rel_x;
	  
	  LineFFTAssert (local_index < _nElements);
	  msg->data [index++] = _line [local_index].re;
	}
      }
    } 

    grid.cb_done.send (msg);
  }
}

#include "PencilFFT.def.h"

