
#ifndef  __PENCIL_FFT_H__
#define  __PENCIL_FFT_H__

#include "charm++.h"
#include "ckcomplex.h"

#include "sfftw.h"

#define LineFFTAssert(x)  //CkAssert(x)
#define __LINEFFT_DEBUG__  0

///
/// The specific phase of the 3D line fft operation
///
enum LineFFTPhase {
  PHASE_X = 0,
  PHASE_Y,
  PHASE_Z
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

  CkCallback              kSpaceCallback;
  LineFFTCompletion       completionId;

  bool                    normalize;
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
  /// \brief Receive an intermediate FFT message
  ///
  void receiveFFTMessage (LineFFTMsg *msg);

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

  /////////////////////////////////////////////////////////////
  ///           End Entry Methods
  /////////////////////////////////////////////////////////////
  
 private:

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

#endif
