#ifndef _CKSTREAM_H
#define _CKSTREAM_H

#include <string.h>
#include <stdio.h>

class _CkOStream {
  private:
    int _isErr;
    int _buflen, _actlen;
    char _obuf[16384];
    char _tbuf[1024];
  public:
    _CkOStream(int isErr=0) { 
      _buflen=16384; 
      _actlen=1;
      _isErr = isErr; 
      _obuf[0] = '\0'; 
    }
    void endl(void) {
      strcat(_obuf, "\n");
      if(_isErr)
        CkError(_obuf);
      else
        CkPrintf(_obuf);
    }

#define _OPSHIFTLEFT(type, format) \
    _CkOStream& operator << (type x) { \
      sprintf(_tbuf, format, x); \
      _actlen += strlen(_tbuf); \
      if(_actlen > _buflen) \
        CmiAbort("Print Buffer Overflow!!\n"); \
      strcat(_obuf, _tbuf); \
      return *this; \
    }

    _OPSHIFTLEFT(int, "%d");
    _OPSHIFTLEFT(unsigned int, "%u");
    _OPSHIFTLEFT(short, "%hd");
    _OPSHIFTLEFT(unsigned short, "%hu");
    _OPSHIFTLEFT(long, "%ld");
    _OPSHIFTLEFT(unsigned long, "%lu");
    _OPSHIFTLEFT(char, "%c");
    _OPSHIFTLEFT(unsigned char, "%u");
    _OPSHIFTLEFT(float, "%f");
    _OPSHIFTLEFT(double, "%lf");
    _OPSHIFTLEFT(const char*, "%s");
    _OPSHIFTLEFT(void*, "%x");
};

static inline _CkOStream& endl(_CkOStream& s)  { s.endl(); return s; }

class _CkOutStream : public _CkOStream {
  public:
    _CkOutStream() : _CkOStream(0) {}
};

class _CkErrStream : public _CkOStream {
  public:
    _CkErrStream() : _CkOStream(1) {}
};

CpvExtern(_CkOutStream*, _ckout);
CpvExtern(_CkErrStream*, _ckerr);

class CkOutStream {
  public:
#define OUTSHIFTLEFT(type) \
  CkOutStream& operator << (type x) { \
    *CpvAccess(_ckout) << x; \
    return *this; \
  }
    OUTSHIFTLEFT(int);
    OUTSHIFTLEFT(unsigned int);
    OUTSHIFTLEFT(short);
    OUTSHIFTLEFT(unsigned short);
    OUTSHIFTLEFT(long);
    OUTSHIFTLEFT(unsigned long);
    OUTSHIFTLEFT(char);
    OUTSHIFTLEFT(unsigned char);
    OUTSHIFTLEFT(float);
    OUTSHIFTLEFT(double);
    OUTSHIFTLEFT(const char*);
    OUTSHIFTLEFT(void*);
};

static inline CkOutStream& endl(CkOutStream& s)  
{ 
  CpvAccess(_ckout)->endl(); return s; 
}

class CkErrStream {
  public:
#define ERRSHIFTLEFT(type) \
  CkErrStream& operator << (type x) { \
    *CpvAccess(_ckerr) << x; \
    return *this; \
  }
    ERRSHIFTLEFT(int);
    ERRSHIFTLEFT(unsigned int);
    ERRSHIFTLEFT(short);
    ERRSHIFTLEFT(unsigned short);
    ERRSHIFTLEFT(long);
    ERRSHIFTLEFT(unsigned long);
    ERRSHIFTLEFT(char);
    ERRSHIFTLEFT(unsigned char);
    ERRSHIFTLEFT(float);
    ERRSHIFTLEFT(double);
    ERRSHIFTLEFT(const char*);
    ERRSHIFTLEFT(void*);
};

static inline CkErrStream& endl(CkErrStream& s)  
{ 
  CpvAccess(_ckerr)->endl(); return s; 
}

extern CkOutStream ckout;
extern CkErrStream ckerr;

class CkInStream {
  public:
#define OPSHIFTRIGHT(type, format) \
    CkInStream& operator >> (type& x) { \
      CkScanf(format, &x); \
      return *this; \
    }

    OPSHIFTRIGHT(int, "%d");
    OPSHIFTRIGHT(unsigned int, "%u");
    OPSHIFTRIGHT(short, "%hd");
    OPSHIFTRIGHT(unsigned short, "%hu");
    OPSHIFTRIGHT(long, "%ld");
    OPSHIFTRIGHT(unsigned long, "%lu");
    OPSHIFTRIGHT(char, "%c");
    OPSHIFTRIGHT(unsigned char, "%u");
    OPSHIFTRIGHT(float, "%f");
    OPSHIFTRIGHT(double, "%lf");

    CkInStream& operator >> (char* x) {
      CkScanf("%s", x);
      return *this;
    }
};

extern CkInStream ckin;

#endif
