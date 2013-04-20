#ifndef _XI_UTIL_H
#define _XI_UTIL_H

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "conv-config.h"

namespace xi {

#if CMK_ISATTY_DECL
#ifdef __cplusplus
extern "C" int isatty (int );
#endif
#endif

#define SZ 1024

class XStr {
  private:
    char *s;
    unsigned int len, blklen;
    void initTo(const char *_s);
    void operator=(const XStr &str); //<- don't use this
  public:
    // MAB: following append methods were earlier private. However,
    // in order to minimize changes to sdag translator, they have been made
    // public. Once the sdag translator is fully embedded in charmxi,
    // they will be made private again.
    void append(const char *_s);
    void append(char c);
    // MAB: the print method is needed for debugging sdag translator.
    // this too will go away later.
    void print(int indent) {
      for (int i=0; i<indent; i++) std::cout << "  ";
      std::cout << get_string();
    }
    /// Appends character c to every line
    void line_append(const char c);
    /// pads with spaces and appends character c to every line. Also converts tabs to spaces.
    void line_append_padding(const char c, int lineWidth=80);
    // Replace all occurences of character "a" in string with character "b"
    void replace (const char a, const char b);
  public:
    XStr();
    XStr(const char *_s);
    XStr(const XStr &_s); //Copy constructor
    ~XStr() { delete[] s; }
    char *get_string(void) const { return s; }
    const char *get_string_const(void) const { return s; }
    // this is to allow XStr to be substituted for CString in
    // structured dagger translator without a lot of changes
    char *charstar(void) const { return get_string(); }
    //This operator allows us to use XStr's interchangably with char *'s:
    operator char *() {return get_string();}
    size_t length() const { return len; }
    //Comparison operators
    int operator==(XStr &s2) const {return 0==strcmp(s,s2.s);}
    int operator!=(XStr &s2) const {return 0!=strcmp(s,s2.s);}
    int operator==(const char *s2) const {return 0==strcmp(s,s2);}
    int operator!=(const char *s2) const {return 0!=strcmp(s,s2);}
    //Addition operator
    XStr operator+ (const XStr &s2) const {XStr ret(*this);ret.append(s2.s); return ret;}
    //Insertion operators
    XStr& operator << (const char *_s) { append(_s); return *this;}
//      XStr& operator << (const string & _s) { append(_s.c_str()); return *this;}
    XStr& operator << (char c) { append(c); return *this;}
    XStr& operator << (int i) ;
    XStr& operator << (const XStr& x) { append(x.get_string_const()); return *this; }
    XStr& operator << (const XStr* x) { append(x->get_string_const()); return *this; }
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

void templateGuardBegin(bool templateOnly, XStr &str);
void templateGuardEnd(XStr &str);

}

#endif
