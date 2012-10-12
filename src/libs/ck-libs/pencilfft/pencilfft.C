
#include "pencilfft.h"

#define PRIORITY_SIZE ((int) sizeof(int)*8)

#define CONST_A  1001
#define CONST_B  17

//#define VERIFY_FFT        1

CmiNodeLock fftw_plan_lock;

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

  _line = new complex[_nElements];
  memset(_line, 0, sizeof(complex) * _nElements);
  
  LineFFTAssert ((_nFFTMessages % CONST_B) != 0);
  
#ifdef VERIFY_FFT 
  for (int count = 0; count < _nElements; count++)
    _line[count] = complex (1.0, 0.0);
#endif  

  CmiLock (fftw_plan_lock);
  _fwdplan = fftw_create_plan(_fftSize, FFTW_FORWARD, FFTW_IN_PLACE | 
			      FFTW_MEASURE | FFTW_USE_WISDOM);
  _bwdplan = fftw_create_plan(_fftSize, FFTW_BACKWARD, FFTW_IN_PLACE | 
			      FFTW_MEASURE | FFTW_USE_WISDOM);
  CmiUnlock (fftw_plan_lock);

  _curFFTMessages  = 0;
  _curGridMessages = 0;
  _nGridMessages   = 0;
  _iter = 0;

#if USE_MANY_TO_MANY
  initialize_many_to_many();
#endif
}

void LineFFTArray::receive_transpose (int           chunk_id, 
				      complex     * data,
				      int           gx,
				      int           gy,
				      int           gz) 
{
  int start = 0;
  int localindex = 0;
  int index = 0;
  switch (_phase) {
  case PHASE_X:
    start = chunk_id * gx;
    for (int z = 0; z < gz; z++) 
      for (int y = 0; y < gy; y++) 
	for (int x = start; x < start + gx; x++) {
	  localindex = _info.sizeX * gy * z + _info.sizeX * y + x;
#if __LINEFFT_DEBUG__
	  LineFFTAssert (localindex < _nElements);
#endif
	  _line [localindex] = data[index++];   
	}    
    break;
    
  case PHASE_Y:  
    start = chunk_id * gy;
    for (int z = 0; z < gz; z++) 
      for (int y = start; y < start+gy; y++) 
	for (int x = 0; x < gx; x++) {
	  localindex = _info.sizeY * gz * x + _info.sizeY * z + y;
#if __LINEFFT_DEBUG__
	  LineFFTAssert (localindex < _nElements);
#endif
	  _line [localindex] = data[index++];   
	}
    break;
    
  case PHASE_Z:
    start = chunk_id * gz;
    for (int z = start; z < start+gz; z++) 
      for (int y = 0; y < gy; y++) 
	for (int x = 0; x < gx; x++) {
	  localindex = _info.sizeZ * gx * y + _info.sizeZ * x + z;
#if __LINEFFT_DEBUG__
	  LineFFTAssert (localindex < _nElements);
#endif
	  _line [localindex] = data[index++];   
	}
    break;

  default:
    CkAbort ("Phase undefined");
    break;
  };    
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

  receive_transpose(msg->chunk_id, msg->data, 
		    _info.grainX, _info.grainY, _info.grainZ);    
  _curFFTMessages ++;
  
  if (_curFFTMessages == _nFFTMessages) {
    _curFFTMessages = 0;
    start_fft (msg->direction);
  }
  
  CkFreeMsg (msg);
}

void LineFFTArray::send_transpose (int           chunk_id, 
				   complex   *   data,
				   int           gx,
				   int           gy,
				   int           gz) 
{
  int start = 0;
  int localindex = 0;
  int index = 0;
  switch (_phase) {
  case PHASE_X:
    start = chunk_id * gx;
    for (int z = 0; z < gz; z++) 
      for (int y = 0; y < gy; y++) 
	for (int x = start; x < start+gx; x++) {
	  localindex = _info.sizeX * gy * z + _info.sizeX * y + x;
#if __LINEFFT_DEBUG__
	  LineFFTAssert (localindex < _nElements);
#endif
	  data[index++] = _line [localindex];
	}        
    break;
    
  case PHASE_Y:
    start = chunk_id * gy;
    for (int z = 0; z < gz; z++) 
      for (int y = start; y < start+gy; y++) 
	for (int x = 0; x < gx; x++) {
	  localindex = _info.sizeY * gz * x + _info.sizeY * z + y;
#if __LINEFFT_DEBUG__
	  LineFFTAssert (localindex < _nElements);
#endif
	  data[index++] = _line [localindex];
	}
    
    break;
      
  case PHASE_Z:
    start = chunk_id * gz;
    for (int z = start; z < start+gz; z++) 
      for (int y = 0; y < gy; y++) 
	for (int x = 0; x < gx; x++) {
	  localindex = _info.sizeZ * gx * y + _info.sizeZ * x + z;
#if __LINEFFT_DEBUG__
	  LineFFTAssert (localindex < _nElements);
#endif
	  data[index++] = _line [localindex];
	}        
    break;
    
  default:
    CkAbort ("Phase undefined");
    break;
  };
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
    int index = 0, localindex =0;
    int chunk_id = (CONST_A*CkMyPe() + CONST_B * count) % _nFFTMessages;

    LineFFTMsg *msg = new (size, sizeof(int) * 8) LineFFTMsg;
    send_transpose(chunk_id, msg->data,
		   _info.grainX, _info.grainY, _info.grainZ);    

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
      LineFFTAssert (dir == FFTW_FORWARD);
      proxy = _info.yProxy;
      msg->phase = PHASE_Y;      
      break;
      
    case PHASE_Y:
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
      LineFFTAssert (dir == FFTW_BACKWARD);
      proxy = _info.yProxy;
      msg->phase = PHASE_Y;      
      break;
      //Default case already verified
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
  _iter = 0;
  start_fft(dir);
}

///
/// \brief Start the 3D fft operation. This can be called externally
/// or intiated when all the grid/fft messages have been received.
///
void LineFFTArray::start_fft (int dir) {

#if __LINEFFT_DEBUG__
  CkPrintf ("[%d, %d, %d, %d] Pencil FFT start fft %d,%d\n", 
	    thisIndex.x, thisIndex.y, _phase, dir, _nElements, _fftSize);
#endif

  fftw_plan plan = NULL;
  
  if (dir == FFTW_FORWARD)
    plan = _fwdplan;
  else
    plan = _bwdplan;
  
  for (int count = 0; count < _nElements; count += _fftSize)
    fftw_one (plan, (fftw_complex *) (_line + count), NULL);

  if (dir == FFTW_FORWARD && _phase == PHASE_Z) { 
    //call_kspace_callback ();    
    start_fft (FFTW_BACKWARD);
    return;
  }

  //Increment iter in backward phase
  if (dir == FFTW_BACKWARD)
    _iter ++;    
  if (dir == FFTW_BACKWARD && _phase == PHASE_X) {
#ifdef VERIFY_FFT
    if ( _info.normalize ) {
      fftw_real scale = 1.0 / (_info.sizeX * _info.sizeY * _info.sizeZ);
      for (int count = 0; count < _nElements; count ++)
	_line [count].re *= scale;
    }
    for (int count = 0; count < _nElements; count++) {
      if (!((_line[count].re <= 1.001)
	    && (_line[count].re >= 0.999)))
	CkPrintf ("[%d, %d] val[%d] = (%5.3f, %5.3f) \n", 
		  thisIndex.x, thisIndex.y,
		  count,
		  _line[count].re, _line[count].im);
    }

    for (int count = 0; count < _nElements; count++)
      _line[count] = complex (1.0, 0.0);
#endif      
    
    if (_iter == _info.numIter) {
      call_donecallback ();      
      return;
    }
    
    dir = FFTW_FORWARD;
    start_fft(dir);
    return; //We starting the next iteration
  }
  

#if USE_MANY_TO_MANY  
  if (_iter > 2) {
    if (_info.grainX == 1 && _info.grainY == 1 && _info.grainZ ==1)
      for (int idx = 0; idx < _nFFTMessages; ++idx) 
	many_to_many_data[idx] = _line[idx];
    else {
      int size = _info.grainX * _info.grainY * _info.grainZ;
      for (int idx = 0; idx < _nFFTMessages; ++idx) 
	send_transpose(idx, many_to_many_data + idx * size,
		       _info.grainX, _info.grainY, _info.grainZ);    
    }
    
    int id = 0;
    if (dir == FFTW_BACKWARD)
      id = 1;
    CmiAssert (tag_s[id] != -1);
    CmiDirect_manytomany_start(handle, tag_s[id]);
  }
  else
#endif
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
    start_fft (FFTW_FORWARD);
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

#if USE_MANY_TO_MANY

void call_ck_cb(void *m) {
  LineFFTCbWrapper *cw = (LineFFTCbWrapper *) m;
  //cw->cb.send(cw->msg);
  cw->me->many_to_many_recvFFT(cw->msg);
}

void LineFFTArray::initialize_many_to_many () {
  bool sfwd = false, sbwd=false;
  bool rfwd = false, rbwd=false;
  int fchunk_id = thisIndex.x;
  int bchunk_id = thisIndex.y;
  
  CProxy_Group sf_proxy;
  if (_phase == PHASE_X) {
    sfwd = true;
    sf_proxy = _info.mapy;
  }
  else if (_phase == PHASE_Y) {
    sfwd = true;    
    sf_proxy = _info.mapz;
  }

  CProxy_Group rf_proxy;
  if (_phase == PHASE_Y) {
    rfwd = true;
    rf_proxy = _info.mapx;
  }
  else if (_phase == PHASE_Z) {
    rfwd = true;
    rf_proxy = _info.mapy;
  }

  CProxy_Group sb_proxy;
  if (_phase == PHASE_Z) {
    sbwd = true;
    sb_proxy = _info.mapy;
  }
  else if (_phase == PHASE_Y) {
    sbwd = true;    
    sb_proxy = _info.mapx;
  }

  CProxy_Group rb_proxy;
  if (_phase == PHASE_Y) {
    rbwd = true;
    rb_proxy = _info.mapz;
  }
  else if (_phase == PHASE_X) {
    rbwd = true;
    rb_proxy = _info.mapy;
  }

  handle = CmiDirect_manytomany_allocate_handle();
  many_to_many_data = new complex[_nElements];
  memset(many_to_many_data, 0, _nElements * sizeof(complex));

  tag_s[0] = tag_s[1] = -1;

  if (sfwd) { //we have forward send
    CmiDirect_manytomany_initialize_sendbase (handle, _phase+1, NULL, NULL, 
					      (char *)many_to_many_data, 
					      _nFFTMessages, fchunk_id);
    tag_s[0] = _phase+1;
  }
  
  if (sbwd) {//we have forward send
    CmiDirect_manytomany_initialize_sendbase (handle, _phase-1+NUM_PHASES, 
					      NULL, NULL, 
					      (char *)many_to_many_data, 
					      _nFFTMessages, bchunk_id);
    tag_s[1] = _phase - 1 + NUM_PHASES;
  }

  //printf ("After send base\n");
    
  int arridx_x, arridx_y;
  int bytes = _info.grainX * _info.grainY * _info.grainZ *sizeof(complex);
  int index = 0;    
  if (sfwd || sbwd) {
    for (index = 0; index < _nFFTMessages; index++) {
      if (sfwd) {
	arridx_x = thisIndex.y;
	arridx_y = index;
	CkArrayMap *amap = (CkArrayMap *) CkLocalBranch(sf_proxy);
	CkArrayIndex2D idx (arridx_x, arridx_y);    
	int pe = amap->procNum (0, idx);    
	LineFFTAssert (pe >= 0);
	LineFFTAssert (pe < CkNumPes());
	
	CmiDirect_manytomany_initialize_send (handle, _phase+1, index, 
					      bytes*index, bytes, pe);        
      }
      if (sbwd) {
	arridx_x = index;
	arridx_y = thisIndex.x;
	CkArrayMap *amap = (CkArrayMap *) CkLocalBranch(sb_proxy);
	CkArrayIndex2D idx (arridx_x, arridx_y);    
	int pe = amap->procNum (0, idx);    
	LineFFTAssert (pe >= 0);
	LineFFTAssert (pe < CkNumPes());
	
	CmiDirect_manytomany_initialize_send (handle, _phase-1+NUM_PHASES,
					      index, bytes*index, bytes, pe); 
      }
    }
  }
  
  //printf("After initialize send\n");

  if (rfwd) {
    CkArrayIndex2D aidx(thisIndex.x, thisIndex.y);
    cbf_recv.cb = CkCallback(CkIndex_LineFFTArray::many_to_many_recvFFT(NULL), 
			     aidx, thisProxy.ckGetArrayID());
    LineFFTCompleteMsg *lmsg = new (PRIORITY_SIZE) LineFFTCompleteMsg;
    lmsg->dir = FFTW_FORWARD;
    cbf_recv.msg = lmsg;
    cbf_recv.me  = this;
    CmiDirect_manytomany_initialize_recvbase (handle, 
					      _phase, 
					      call_ck_cb, 
					      &cbf_recv, 
					      (char *) many_to_many_data, 
					      _nFFTMessages,
					      -1); //no local messages
  }
  
  if (rbwd) {
    CkArrayIndex2D aidx(thisIndex.x, thisIndex.y);
    cbb_recv.cb = CkCallback(CkIndex_LineFFTArray::many_to_many_recvFFT(NULL), 
			     aidx, thisProxy.ckGetArrayID());
    
    LineFFTCompleteMsg *lmsg = new (PRIORITY_SIZE) LineFFTCompleteMsg;
    lmsg->dir = FFTW_BACKWARD;
    cbb_recv.msg = lmsg;
    cbb_recv.me  = this;

    CmiDirect_manytomany_initialize_recvbase (handle, 
					      _phase + NUM_PHASES, 
					      call_ck_cb, 
					      &cbb_recv, 
					      (char *) many_to_many_data, 
					      _nFFTMessages,
					      -1); //no local messages
  }

  if (rfwd || rbwd) {
    for (index = 0; index < _nFFTMessages; index++) {
      if (rfwd) {
	arridx_x = index; 
	arridx_y = thisIndex.x; 
	CkArrayMap *amap = (CkArrayMap *) CkLocalBranch(rf_proxy);
	CkArrayIndex2D idx (arridx_x, arridx_y);    
	int pe = amap->procNum (0, idx);    
	LineFFTAssert (pe >= 0);
	LineFFTAssert (pe < CkNumPes());         
	
	CmiDirect_manytomany_initialize_recv (handle, _phase, index, 
					      bytes*index, bytes, pe);        
      }
      if (rbwd) {
	arridx_x = thisIndex.y;
	arridx_y = index; 
	CkArrayMap *amap = (CkArrayMap *) CkLocalBranch(rb_proxy);
	CkArrayIndex2D idx (arridx_x, arridx_y);    
	int pe = amap->procNum (0, idx);    
	LineFFTAssert (pe >= 0);
	LineFFTAssert (pe < CkNumPes());         
	
	CmiDirect_manytomany_initialize_recv (handle, _phase+NUM_PHASES, 
					      index, bytes*index, bytes, pe);
      }
    }    
  }  
}

void LineFFTArray::many_to_many_recvFFT (LineFFTCompleteMsg *msg) {
  //printf("in many_to_many_recvFFT\n");
  if (_info.grainX == 1 && _info.grainY == 1 && _info.grainZ == 1) {
    for (int idx = 0; idx < _nFFTMessages; ++idx) 
      _line[idx] = many_to_many_data[idx];
  }
  else {
    int size = _info.grainX *_info.grainY *_info.grainZ;    
    for (int idx = 0; idx < _nFFTMessages; ++idx )
      receive_transpose(idx, many_to_many_data + idx * size,
			_info.grainX, _info.grainY, _info.grainZ);    
  }

  start_fft(msg->dir);
}

#endif

#include "PencilFFT.def.h"

