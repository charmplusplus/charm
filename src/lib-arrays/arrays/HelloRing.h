#include "charm++.h"
#include "HelloRing.top.h"
#include "ArrayElement.h"
#include "Array1D.h"
#include "HelloMessage.h"

class TokenMessage : public comm_object
{
  int dummy;
};

class HelloRing : public ArrayElement
{
public:
  HelloRing(ArrayElementCreateMessage *msg);
  HelloRing(ArrayElementMigrateMessage *msg);
  void hello(HelloMessage *msg);
  int packsize(void);
  void pack(void *buf);

private:
  double x;
};
