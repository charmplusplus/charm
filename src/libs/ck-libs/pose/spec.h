// File: spec.h
#ifndef SPEC_H
#define SPEC_H

class spec : public opt3 {
public:
  spec();
  virtual void Step();              // single forward execution step
};

#endif
