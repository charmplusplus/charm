/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _XI_UTIL_H
#define _XI_UTIL_H

#include <string.h>
//Jay, this include may cause problems--
//not all compilers support ISO C++ include files (OSL, 4/3/2000)
//  #include <string>
//  using std::string;//<- and not all compilers support namespaces

#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>

/*HACK: Define bool as int; false as 0; true as 1.  
This lets us compile on machines without a "bool" type.
*/
#ifndef bool
# define bool int
# define false 0
# define true 1
#endif

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
    //Comparison operators
    int operator==(const XStr &s2) {return 0==strcmp(s,s2.s);}
    int operator!=(const XStr &s2) {return 0!=strcmp(s,s2.s);}
    int operator==(const char *s2) {return 0==strcmp(s,s2);}
    int operator!=(const char *s2) {return 0!=strcmp(s,s2);}
    //Addition operator
    XStr operator+ (const XStr &s2) const {XStr ret(*this);ret.append(s2.s); return ret;}
    //Insertion operators
    XStr& operator << (const char *_s) { append(_s); return *this;}
//      XStr& operator << (const string & _s) { append(_s.c_str()); return *this;}
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
    //This lets us cast printables to XStr
    operator XStr () {XStr ret;print(ret);return ret;}
    //These let us stream Printables to XStr.
    friend XStr & operator << (XStr &str,Printable &p) {p.print(str);return str;}
    friend XStr & operator << (XStr &str,Printable *p) {p->print(str);return str;}
};

#endif
