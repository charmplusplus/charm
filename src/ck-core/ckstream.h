
#ifndef _CKSTREAM_H
#define _CKSTREAM_H

#include <string.h>
#include <stdio.h>

#define BUF_MAXLEN  16384
#define TBUF_MAXLEN   128

#if defined(_WIN32) && ! defined(__CYGWIN__)
#define snprintf _snprintf
#endif

class _CkOStream {
  private:
    int _isErr;
    size_t _buflen, _actlen;
    char _obuf[BUF_MAXLEN];    /* stores a line of text */
    char _tbuf[TBUF_MAXLEN];   /* used for formatting ints and things */
    void output(const char *str) {
      _actlen += strlen(str);
      if(_actlen > _buflen)
        CmiAbort("Print Buffer Overflow!!\n");
      strcat(_obuf, str); 
    }
    
  public:
    _CkOStream(int isErr=0) { 
      _buflen=BUF_MAXLEN; 
      _actlen=1;
      _isErr = isErr; 
      _obuf[0] = '\0'; 
    }
    _CkOStream& endl(void) {
      strcat(_obuf, "\n");
      if(_isErr)
        CkError("%s", _obuf);
      else
        CkPrintf("%s", _obuf);
      _obuf[0] = '\0'; 
      _actlen=1;
      return *this;
    }

    _CkOStream& operator << (_CkOStream& (*f)(_CkOStream &)) {
      return f(*this);
    }
#define _OPSHIFTLEFT(type, format) \
    _CkOStream& operator << (type x) { \
      int ret = snprintf(_tbuf, TBUF_MAXLEN, format, (type) x); \
      if (ret >= TBUF_MAXLEN) CmiPrintf("Warning: CkStream tbuf overflow!\n"); \
      output(_tbuf); \
      return *this; \
    }

    _OPSHIFTLEFT(int, "%d");
    _OPSHIFTLEFT(unsigned int, "%u");
    _OPSHIFTLEFT(short, "%hd");
    _OPSHIFTLEFT(unsigned short, "%hu");
    _OPSHIFTLEFT(long, "%ld");
    _OPSHIFTLEFT(unsigned long, "%lu");
#if CMK_LONG_LONG_DEFINED
    _OPSHIFTLEFT(long long, "%lld");
    _OPSHIFTLEFT(unsigned long long, "%llu");
#endif
    _OPSHIFTLEFT(char, "%c");
    _OPSHIFTLEFT(unsigned char, "%u");
    _OPSHIFTLEFT(float, "%f");
    _OPSHIFTLEFT(double, "%f");  // Floats and doubles are identical for printf
    _OPSHIFTLEFT(void*, "%p");

    /* Avoid _tbuf overflow, by outputting strings directly... */
    _CkOStream& operator << (const char *str) {
      output(str);
      return *this;
    }
};

static inline _CkOStream& endl(_CkOStream& s)  { return s.endl(); }

class _CkOutStream : public _CkOStream {
  public:
    _CkOutStream() : _CkOStream(0) {}
};

class _CkErrStream : public _CkOStream {
  public:
    _CkErrStream() : _CkOStream(1) {}
};

CkpvExtern(_CkOutStream*, _ckout);
CkpvExtern(_CkErrStream*, _ckerr);

class CkOStream {
 public:
  virtual ~CkOStream() {}
  virtual CkOStream& operator << (_CkOStream& (*f)(_CkOStream &)) = 0;
#define SHIFTLEFT(type) \
  virtual CkOStream& operator << (type x) = 0

  SHIFTLEFT(int);
  SHIFTLEFT(unsigned int);
  SHIFTLEFT(short);
  SHIFTLEFT(unsigned short);
  SHIFTLEFT(long);
  SHIFTLEFT(unsigned long);
#if CMK_LONG_LONG_DEFINED
  SHIFTLEFT(long long);
  SHIFTLEFT(unsigned long long);
#endif
  SHIFTLEFT(char);
  SHIFTLEFT(unsigned char);
  SHIFTLEFT(float);
  SHIFTLEFT(double);
  SHIFTLEFT(const char*);
  SHIFTLEFT(void*);
};

class CkOutStream : public CkOStream {
  public:
  CkOutStream& operator << (_CkOStream& (*f)(_CkOStream &)) {
    f(*CkpvAccess(_ckout));
    return *this;
  }
#define OUTSHIFTLEFT(type) \
  CkOutStream& operator << (type x) { \
    *CkpvAccess(_ckout) << x; \
    return *this; \
  }
    OUTSHIFTLEFT(int);
    OUTSHIFTLEFT(unsigned int);
    OUTSHIFTLEFT(short);
    OUTSHIFTLEFT(unsigned short);
    OUTSHIFTLEFT(long);
    OUTSHIFTLEFT(unsigned long);
#if CMK_LONG_LONG_DEFINED
    OUTSHIFTLEFT(long long);
    OUTSHIFTLEFT(unsigned long long);
#endif
    OUTSHIFTLEFT(char);
    OUTSHIFTLEFT(unsigned char);
    OUTSHIFTLEFT(float);
    OUTSHIFTLEFT(double);
    OUTSHIFTLEFT(const char*);
    OUTSHIFTLEFT(void*);
};

class CkErrStream : public CkOStream {
  public:
  CkErrStream& operator << (_CkOStream& (*f)(_CkOStream &)) {
    f(*CkpvAccess(_ckerr));
    return *this;
  }
#define ERRSHIFTLEFT(type) \
  CkErrStream& operator << (type x) { \
    *CkpvAccess(_ckerr) << x; \
    return *this; \
  }
    ERRSHIFTLEFT(int);
    ERRSHIFTLEFT(unsigned int);
    ERRSHIFTLEFT(short);
    ERRSHIFTLEFT(unsigned short);
    ERRSHIFTLEFT(long);
    ERRSHIFTLEFT(unsigned long);
#if CMK_LONG_LONG_DEFINED
    ERRSHIFTLEFT(long long);
    ERRSHIFTLEFT(unsigned long long);
#endif
    ERRSHIFTLEFT(char);
    ERRSHIFTLEFT(unsigned char);
    ERRSHIFTLEFT(float);
    ERRSHIFTLEFT(double);
    ERRSHIFTLEFT(const char*);
    ERRSHIFTLEFT(void*);
};

extern CkOutStream ckout;
extern CkErrStream ckerr;

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
#if CMK_LONG_LONG_DEFINED
    OPSHIFTRIGHT(long long, "%lld");
    OPSHIFTRIGHT(unsigned long long, "%llu");
#endif
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
