// File: opt3.h
#ifndef OPT3_H
#define OPT3_H

class opt3 : public opt {
 protected:
  int timeLeash;
public:
  opt3();
  virtual void Step();              // single forward execution step
};

#endif
