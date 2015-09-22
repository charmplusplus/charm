#ifndef _XI_UTIL_H
#define _XI_UTIL_H

#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "conv-config.h"

#define XLAT_ERROR(...) \
  do {                                            \
    if (xi::num_errors++ == xi::MAX_NUM_ERRORS) { \
      exit(1);                                    \
    } else {                                      \
      pretty_msg("error", __VA_ARGS__);           \
    }                                             \
  } while (0)

#define XLAT_ERROR_NOCOL(str,line) XLAT_ERROR((str), -1, -1, (line), (line))

#define XLAT_NOTE(str,line) pretty_msg("note", (str), -1, -1, (line), (line))

extern unsigned int lineno;

namespace xi {

extern void pretty_msg(std::string type, std::string msg,
                       int first_col=-1, int last_col=-1,
                       int first_line=-1, int last_line=-1);

extern const int MAX_NUM_ERRORS;
extern int num_errors;

extern std::vector<std::string> inputBuffer;

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
    void clear();
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
  virtual ~Printable(){}
  friend XStr & operator << (XStr &str,Printable &p) {p.print(str);return str;}
  friend XStr & operator << (XStr &str,Printable *p) {p->print(str);return str;}
};

void templateGuardBegin(bool templateOnly, XStr &str);
void templateGuardEnd(XStr &str);

inline void indentBy(XStr& s, int num) {
    for (int i = 0; i < num; i++) s << "  ";
}

class TVarList;
XStr generateTemplateSpec(TVarList* tspec, bool printDefault = true);

typedef enum {
  forAll=0,forIndividual=1,forSection=2,forPython=3,forIndex=-1
} forWhom;

const char *forWhomStr(forWhom w);

// FIXME: this same function is used for both syntax error messages as well as
//        e.g. code generation errors
void die(const char *why, int line=-1);

char* fortranify(const char *s, const char *suff1="", const char *suff2="", const char *suff3="");

void templateGuardBegin(bool templateOnly, XStr &str);
void templateGuardEnd(XStr &str);

std::string addLineNumbers(char *str, const char *filename);
extern void sanitizeComments(std::string &code);
extern void sanitizeStrings(std::string &code);
extern void desanitizeCode(std::string &code);

}   // namespace xi

namespace Prefix {

extern const char *Proxy;
extern const char *ProxyElement;
extern const char *ProxySection;
extern const char *Message;
extern const char *Index;
extern const char *Python;

}   // namespace Prefix

#endif
