#ifndef _XI_UTIL_H
#define _XI_UTIL_H

#include <string.h>
#include <iostream.h>
#include <stdlib.h>

#define SZ 1024

class XStr {
  private:
    char *s;
    unsigned int len, blklen;
    void append(const char *_s);
    void append(char c);
  public:
    XStr();
    XStr(const char *_s);
    ~XStr() { delete[] s; }
    char *get_string(void) { return s; }
    XStr& operator << (const char *_s) { append(_s); return *this;}
    XStr& operator << (char c) { append(c); return *this;}
    XStr& operator << (XStr& x) { append(x.get_string()); return *this; }
    void spew(const char*b, const char *a1 = 0, const char *a2 = 0, 
              const char *a3 = 0, const char *a4 = 0, const char *a5 = 0);
};

#define endx "\n"

static inline const char *chare_prefix(void) { return "CProxy_"; }

static inline const char *group_prefix(void) { return "CProxy_"; }

static inline const char *array_prefix(void) { return "CProxy_"; }

static inline const char *msg_prefix(void) { return "CMessage_"; }

class Printable {
  public:
    virtual void print(XStr& str) = 0;
};

#endif
