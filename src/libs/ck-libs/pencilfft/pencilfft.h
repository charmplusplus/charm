
#ifndef  __PENCIL_FFT_H__
#define  __PENCIL_FFT_H__

#include "charm++.h"
#include "ckcomplex.h"
#include "sfftw.h"

//#define USE_MANY_TO_MANY  1

#if USE_MANY_TO_MANY
#include "cmidirectmanytomany.h"
#endif

#define LineFFTAssert(x)  //CkAssert(x)
#define __LINEFFT_DEBUG__  0

///
/// The specific phase of the 3D line fft operation
///
enum LineFFTPhase {
  PHASE_X = 0,
  PHASE_Y,
  PHASE_Z,
  NUM_PHASES
};


enum LineFFTCompletion {
  ARRAY_REDUCTION = 0,
  SEND_GRID,
};


///
/// \brief The Grid structures which describes the points being sent 
///
struct LineFFTGrid {
  short  x0;
  short  xsize;
  short  y0;
  short  ysize;
  short  z0;
  short  zsize;  

  short  idx_x;    //The x-coordinate of the pencil array element
  short  idx_y;    //The y-coordinate of the pencil array element

  CkCallback cb_done;   //Callback to call once the operation has been
			//finished
};

PUPbytes (LineFFTGrid)

struct LineFFTInfo;

#include "PencilFFT.decl.h"

///
/// Startup consturction info for the line fft array element
///
struct LineFFTInfo {
  int sizeX, sizeY, sizeZ;    //FFT grid dimensions
  int grainX, grainY, grainZ; //FFT pencil partition sizes
  
  ///
  /// \brief Proxy for X, Y, Z chare arrays. This will allow each of
  /// them to be a different chare array possibly mapped to different a
  /// processor
  ///
  CProxy_LineFFTArray     xProxy;
  CProxy_LineFFTArray     yProxy; 
  CProxy_LineFFTArray     zProxy;

  CProxy_PencilMapX       mapx;
  CProxy_PencilMapY       mapy;
  CProxy_PencilMapZ       mapz;

  CkCallback              kSpaceCallback;
  LineFFTCompletion       completionId;

  bool                    normalize;
  int                     numIter;
};

PUPbytes (LineFFTInfo)

///////////////////////////////////////////////////////////////
//             The Pencil FFT Array Class
//
//  Flow of Control :
//   Client -> receiveGridMessage() > startFFT(X, Forward) -> 
//   sendFFTMessages() -> startFFT(Y, Forward) -> 
//   startFFT(Z, Forward) -> startFFT(Z, Backward) -> 
//   startFFT(Y, Backward) -> startFFT(X, Backward) -> 
//   sendGridMessages () -> Client
//
///////////////////////////////////////////////////////////////

class LineFFTMsg : public CMessage_LineFFTMsg {
 public:
  LineFFTPhase       phase;     //the phase of the operation
  int                direction; //direction of the fft
  int                chunk_id;  //the id of the chunk or a pencil
  
  ///
  /// \brief data always stored in an XYZ order
  ///
  complex           * data;  
};

class LineFFTGridMsg : public CMessage_LineFFTGridMsg {
 public:
  LineFFTGrid         grid;
  fftw_real         * data;  
};

class LineFFTCompleteMsg :  public CMessage_LineFFTCompleteMsg {
 public:
  int     dir;
};

class LineFFTArray;

struct LineFFTCbWrapper {
  CkCallback                    cb;
  LineFFTCompleteMsg          * msg;
  LineFFTArray                * me;
};

PUPbytes (LineFFTCbWrapper)

///
/// Main Pencil FFT Chare Array
///
class LineFFTArray : public CBase_LineFFTArray {
  ///
  /// \brief The fft points. The order of the points depends on the
  /// phase. For example in Phase_X they will be stored in an XYZ order.
  ///
  complex *_line;
  
  ///Forward and backward fftw plans
  fftw_plan _fwdplan, _bwdplan;
  
  ///
  /// The FFT info structure
  ///
  LineFFTInfo   _info;
  
  LineFFTPhase  _phase;  //X, Y or Z chare

  ///\brief Iteration count
  int         _iter;

  ///
  /// \brief the current number of grid messages received so far
  ///
  int           _curGridMessages;

  ///
  /// \brief number of messssages received
  ///
  int           _curFFTMessages;

  ///
  /// \brief number of fft points in this thick pencil
  ///
  int           _nElements;

  ///
  /// \brief size of the fft
  ///
  int           _fftSize;

  int           _nFFTMessages;

  ///
  /// \brief Number of grid messages to expect before starting the FFT
  ///
  int           _nGridMessages;

  ///
  /// \brief List of grids to send back to the the client
  ///
  LineFFTGrid  * _gridList;

 public:

  /////////////////////////////////////////////////////
  ///            Entry Methods
  /////////////////////////////////////////////////////

  ///
  /// \brief Add loadbalancing and migration later
  ///
  LineFFTArray (CkMigrateMessage *m) {}

  /// 
  /// \brief The default constructor
  ///
  LineFFTArray () {
    _fwdplan = _bwdplan =NULL;
    //_id = -1;
    _line = NULL;
    _curFFTMessages = 0;
    _curGridMessages = 0;
    _nGridMessages = 0;
  }

  ///
  /// \brief the real constructor
  ///
  LineFFTArray (LineFFTInfo &info, int phase);
  
  ~LineFFTArray () {
    delete [] _line;
  }

  ///
  /// \brief Receive data points to be fft'ed. This will happen in the
  /// first and last phases
  ///
  void receiveGridMessage (LineFFTGridMsg   *msg);

  ///
  /// \brief transpose an intermediate FFT message
  ///
  void receive_transpose (int id, complex *data, int, int, int);

  ///
  /// \brief Receive an intermediate FFT message
  ///
  void receiveFFTMessage (LineFFTMsg *msg);

  ///
  /// \brief Start the 3D fft operation. This can be called externally
  /// or intiated when all the grid/fft messages have been received.
  ///
  void start_fft (int direction = FFTW_FORWARD);

  ///
  /// \brief Start the 3D fft operation. This can be called externally
  /// or intiated when all the grid/fft messages have been received.
  ///
  void startFFT (int direction = FFTW_FORWARD);

  ///
  /// \brief Set the number of grid messages to expect
  ///
  void setNumGridMessages (int n, CkCallback &cb) {
    _nGridMessages = n;
    
#if __LINEFFT_DEBUG__
    CkPrintf ("[%d, %d] Received ngridmsgs msg \n", thisIndex.x, thisIndex.y);
#endif
    _gridList = new LineFFTGrid [n];
    
    int x =0;
    contribute (sizeof (int), &x, CkReduction::sum_int, cb);
  }

#if USE_MANY_TO_MANY 
  void             * handle;   //many to many handle
  int                tag_s[2]; //fwd and bwd send tags
  complex          * many_to_many_data;
  LineFFTCbWrapper   cbf_recv; //fwd recv completion
  LineFFTCbWrapper   cbb_recv; //bwd recv completion

  void initialize_many_to_many ();

  ///
  /// \brief many to many recv completion
  ///
  void many_to_many_recvFFT(LineFFTCompleteMsg *msg);
#else
  void many_to_many_recvFFT(LineFFTCompleteMsg *msg) {
    //CmiAbort();
  }
#endif

  /////////////////////////////////////////////////////////////
  ///           End Entry Methods
  /////////////////////////////////////////////////////////////
  
 private:

  ///
  /// \brief Send the fft messages for the next few phases
  /// 
  void send_transpose (int id, complex *data, int, int, int);

  ///
  /// \brief Send the fft messages for the next few phases
  /// 
  void sendFFTMessages (int dir);

  ///
  /// \brief Send the grids back to the callers
  ///
  void sendGridMessages();
  
  ///
  ///  \brief Call the application done callback
  ///
  void call_donecallback() {
    LineFFTAssert (_phase == PHASE_X);

#if __LINEFFT_DEBUG__    
    CkPrintf ("[%d, %d] In Call Done Callback\n", thisIndex.x, thisIndex.y);
#endif
    if (_info.completionId == ARRAY_REDUCTION) {
      int x =0;
      contribute (sizeof (int), &x, CkReduction::sum_int);
    }    
    else 
      sendGridMessages ();
  }
};   // -- End class LineFFTArray

extern CmiNodeLock fftw_plan_lock;

class PencilMapX : public CBase_PencilMapX
{
  LineFFTInfo   _info;
  int         * _mapcache;
  int           _sizeY;

 public:
  PencilMapX(LineFFTInfo  &info) { 
    _info = info;         
    if (CmiMyRank() == 0)
      if (fftw_plan_lock == NULL)
	fftw_plan_lock = CmiCreateLock();
    
    initialize();
  }
  PencilMapX(CkMigrateMessage *m){}

  void initialize () {    
    double pe = 0.0;
    int y = 0, z = 0;    
    int nelem = (_info.sizeY * _info.sizeZ)/
      (_info.grainY * _info.grainZ);    
    if ((CkNumPes() % nelem) != 0)
      nelem ++;
    double stride = (int) ((1.0 *CkNumPes())/ nelem);
#if USE_MANY_TO_MANY
    CmiAssert (stride >= 1.0);
#endif
    _mapcache = new int[nelem];
    memset (_mapcache, 0, sizeof(int)*nelem);
    _sizeY = _info.sizeY/_info.grainY;	

    int idx = 0;
    for (pe = 0.0, z = 0; z < (_info.sizeZ)/(_info.grainZ); z ++) {
      for (y = 0; y < (_info.sizeY)/(_info.grainY); y++) {
	if(pe >= CkNumPes()) {
	  pe = pe - CkNumPes();	
	  if ((int)pe == 0)
	    pe++;
	}
	idx = z * _sizeY + y;	
	_mapcache[idx] = (int)pe;
	pe +=  stride;
      }
    }
    //if (CkMyPe() == 0)
    //printf("X Last allocation to %d\n", (int)pe);    
  }

  int procNum(int, const CkArrayIndex &idx) {
    CkArrayIndex2D idx2d = *(CkArrayIndex2D *) &idx;    
    int y = idx2d.index[0];
    int z = idx2d.index[1];
    int id = z * _sizeY + y;

    return _mapcache[id];
  }
};


class PencilMapY : public CBase_PencilMapY
{
  LineFFTInfo   _info;
  int         * _mapcache;
  int           _sizeZ;

 public:
  PencilMapY(LineFFTInfo  &info) { 
    _info = info;         
    if (CmiMyRank() == 0)
      if (fftw_plan_lock == NULL)
	fftw_plan_lock = CmiCreateLock();
    
    initialize();
  }
  PencilMapY(CkMigrateMessage *m){}

  void initialize () {    
    double pe = 0.0;
    int z = 0, x = 0;    
    int nelem = (_info.sizeZ * _info.sizeX)/
      (_info.grainZ * _info.grainX);    
    if ((CkNumPes() % nelem) != 0)
      nelem ++;
    double stride = (int)((1.0 *CkNumPes())/ nelem);
#if USE_MANY_TO_MANY
    CmiAssert (stride >= 1.0);
#endif
    _mapcache = new int[nelem];
    memset (_mapcache, 0, sizeof(int)*nelem);
    _sizeZ = _info.sizeZ/_info.grainZ;	

    int idx = 0;
    for (pe = 1.0, x = 0; x < (_info.sizeX)/(_info.grainX); x ++) {
      for (z = 0; z < (_info.sizeZ)/(_info.grainZ); z++) {
	if(pe >= CkNumPes()) {
	  pe = pe - CkNumPes();	
	  if ((int)pe == 1)
	    pe++;
	}
	idx = x * _sizeZ + z;	
	_mapcache[idx] = (int)pe;
	pe +=  stride;
      }
    }

    //if (CkMyPe() == 0)
    //printf("Y Last allocation to %d\n", (int)pe);
  }

  int procNum(int, const CkArrayIndex &idx) {
    CkArrayIndex2D idx2d = *(CkArrayIndex2D *) &idx;    
    int z = idx2d.index[0];
    int x = idx2d.index[1];
    int id = x * _sizeZ + z;

    return _mapcache[id];
  }
};

////////////////////////////////////////////

class PencilMapZ : public CBase_PencilMapZ
{
  LineFFTInfo   _info;
  int         * _mapcache;
  int           _sizeX;

 public:
  PencilMapZ(LineFFTInfo  &info) { 
    _info = info;         
    if (CmiMyRank() == 0)
      if (fftw_plan_lock == NULL)
	fftw_plan_lock = CmiCreateLock();
    
    initialize();
  }
  PencilMapZ(CkMigrateMessage *m){}

  void initialize () {    
    double pe = 0.0;
    int x = 0, y = 0;    
    int nelem = (_info.sizeX * _info.sizeY)/
      (_info.grainX * _info.grainY);    
    if ((CkNumPes() % nelem) != 0)
      nelem ++;
    double stride = (int)((1.0 *CkNumPes())/ nelem);
#if USE_MANY_TO_MANY
    CmiAssert (stride >= 1.0);
#endif
    _mapcache = new int[nelem];    
    memset (_mapcache, 0, sizeof(int)*nelem);
    _sizeX = _info.sizeX/_info.grainX;	

    int idx = 0;
    for (pe = 0.0, x = 0; x < (_info.sizeX)/(_info.grainX); x ++) {
      for (y = 0; y < (_info.sizeY)/(_info.grainY); y++) {
	if(pe >= CkNumPes()) {
	  pe = pe - CkNumPes();	
	  if((int) pe == 0)
	    pe++;
	}
	idx = y * _sizeX + x;	
	_mapcache[idx] = (int)pe;
	pe +=  stride;
      }
    }
    //if (CkMyPe() == 0)
    //printf("Z Last allocation to %d\n", (int)pe);
  }

  int procNum(int, const CkArrayIndex &idx) {
    CkArrayIndex2D idx2d = *(CkArrayIndex2D *) &idx;    
    int x = idx2d.index[0];
    int y = idx2d.index[1];
    int id = y * _sizeX + x;

    return _mapcache[id];
  }
};

#endif
