// File: adapt2.h
#ifndef ADAPT2_H
#define ADAPT2_H

class adapt2 : public opt3 {
 protected:
  int eventLeash;  // leash on the _quantity_ of events to execute in a Step
 public:
  adapt2();
  virtual void Step();  // forward execution
};

#endif
