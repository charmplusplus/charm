// File: adapt.h
// Module for adaptive optimistic synchronization strategy class
// Last Modified: 02.26.03 by Terry L. Wilmarth

#ifndef ADAPT_H
#define ADAPT_H

class adapt : public opt {
public:
  adapt();
  virtual void Step();              // single forward execution step
};

#endif
