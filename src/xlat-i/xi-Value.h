#ifndef _VALUE_H
#define _VALUE_H

#include "xi-util.h"

namespace xi {

class XStr;

class Value : public Printable {
 private:
  int factor;
  const char *val;
 public:
  Value(const char *s);
  void print(XStr& str);
  int getIntVal(void);
};

class ValueList : public Printable {
 private:
  Value *val;
  ValueList *next;
 public:
  ValueList(Value* v, ValueList* n=0);
  void print(XStr& str);
  void printValue(XStr& str);
  void printValueProduct(XStr& str);
  void printZeros(XStr& str);
};

}

#endif  // ifndef _VALUE_H
