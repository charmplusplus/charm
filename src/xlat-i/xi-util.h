/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _XI_UTIL_H
#define _XI_UTIL_H

#include <string.h>
#include <string>
#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>

#define SZ 1024

class XStr {
  private:
    char *s;
    unsigned int len, blklen;
    void append(const char *_s);
    void append(char c);
    void initTo(const char *_s);
  public:
    XStr();
    XStr(const char *_s);
    XStr(const XStr &_s); //Copy constructor
    ~XStr() { delete[] s; }
    char *get_string(void) { return s; }
    const char *get_string_const(void) const { return s; }
    //This operator allows us to use XStr's interchangably with char *'s:
    operator char *() {return get_string();}
    XStr& operator << (const char *_s) { append(_s); return *this;}
    XStr& operator << (const string & _s) { append(_s.c_str()); return *this;}
    XStr& operator << (char c) { append(c); return *this;}
    XStr& operator << (int i) ;
    XStr& operator << (const XStr& x) { append(x.get_string_const()); return *this; }
    void spew(const char*b, const char *a1 = 0, const char *a2 = 0, 
              const char *a3 = 0, const char *a4 = 0, const char *a5 = 0);
};

#define endx "\n"

class Printable {
  public:
    virtual void print(XStr& str) = 0;
    //These let us stream Printables to XStr.
    friend XStr & operator << (XStr &str,Printable &p) {p.print(str);return str;}
    friend XStr & operator << (XStr &str,Printable *p) {p->print(str);return str;}
};

#endif
