// File: adapt.h
#ifndef ADAPT_H
#define ADAPT_H

class adapt : public opt3 {
public:
  adapt();
  virtual void Step();              // single forward execution step
};

#endif
