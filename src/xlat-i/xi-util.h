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
    void append(char *_s);
    void append(char c);
  public:
    XStr();
    XStr(char *_s);
    ~XStr() { delete[] s; }
    char *get_string(void) { return s; }
    XStr& operator << (char *_s) { append(_s); return *this;}
    XStr& operator << (char c) { append(c); return *this;}
    XStr& operator << (XStr& x) { append(x.get_string()); return *this; }
    void spew(const char*b, char *a1 = 0, char *a2 = 0, char *a3 = 0,
              char *a4 = 0, char *a5 = 0);
};

#define endx "\n"

static inline char *chare_prefix(void) { return "CProxy_"; }

static inline char *group_prefix(void) { return "CProxy_"; }

static inline char *array_prefix(void) { return "CProxy_"; }

static inline char *msg_prefix(void) { return "CMessage_"; }

class Printable {
  public:
    virtual void print(XStr& str) = 0;
};

#endif
