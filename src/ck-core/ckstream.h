/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CKSTREAM_H
#define _CKSTREAM_H

// Because different people and platforms prefer
//  different versions of the standard streambuf and
//  ostream classes, they are protected these via typedefs.
//  Eventually everybody should use the std:: versions.
#if CMK_STL_USE_DOT_H || defined(CK_IOSTREAM_DOT_H)
/* Weird pre-ISO nonstandard headers */
# include <iostream.h>
  typedef streambuf Ck_std_streambuf;
  typedef ostream Ck_std_ostream;
#else
/* Normal ISO standard headers */
# include <iostream>
  typedef std::streambuf Ck_std_streambuf;
  typedef std::ostream Ck_std_ostream;
#endif

/** 
 * This std::streambuf buffers up characters for
 * output via CkPrintf or CkError.
 */
class CkOStreamBuf : public Ck_std_streambuf {
public:
  CkOStreamBuf (int isErr_) { 
    isErr=isErr_; 
    resetMyBuffer();
  }
  ~CkOStreamBuf() {sync();}
  
protected:
  int isErr;
  enum {BUFSIZE=1024};
  char buf[BUFSIZE+1]; // My output buffer
  
  /// The one true output routine: write these n characters.
  ///  buf is always zero-terminated.
  void myWrite(const char *buf,int n);
  
  /// Set up the streambuf with my output buffer.
  void resetMyBuffer(void) {
    setp(buf,buf+BUFSIZE);
  }

  /// std::streambuf routine: write buffer to output
  int sync (void);
  /// std::streambuf routine: buffer is full
  int overflow (int ch);
};

/// A std::ostream that sends its output to CkPrintf.
class CkOutStream : public Ck_std_ostream {
	CkOStreamBuf buf;
public:
	CkOutStream() :Ck_std_ostream(&buf), buf(0) {}
};

/// A std::ostream that sends its output to CkError.
class CkErrStream : public Ck_std_ostream {
	CkOStreamBuf buf;
public:
	CkErrStream() :Ck_std_ostream(&buf), buf(1) {}
};

// For SMP safety, keep a separate output stream per thread:
CpvExtern(CkOutStream,_ckout);
CpvExtern(CkErrStream,_ckerr);
#define ckout CpvAccess(_ckout)
#define ckerr CpvAccess(_ckerr)



/**
 * A silly, silly replacement for std::istream.
 *  This should probably just be removed, since it's nonstandard.
 */
class CkInStream {
  public:
#define OPSHIFTRIGHT(type, format) \
    CkInStream& operator >> (type& x) { \
      CkScanf(format, (type *)&x); \
      return *this; \
    }

    OPSHIFTRIGHT(int, "%d");
    OPSHIFTRIGHT(unsigned int, "%u");
    OPSHIFTRIGHT(short, "%hd");
    OPSHIFTRIGHT(unsigned short, "%hu");
    OPSHIFTRIGHT(long, "%ld");
    OPSHIFTRIGHT(unsigned long, "%lu");
    OPSHIFTRIGHT(char, "%c");
    OPSHIFTRIGHT(unsigned char, "%c");
    OPSHIFTRIGHT(float, "%f");
    OPSHIFTRIGHT(double, "%lf");

    CkInStream& operator >> (char* x) {
      CkScanf("%s", x);
      return *this;
    }
};
extern CkInStream ckin;

#endif
