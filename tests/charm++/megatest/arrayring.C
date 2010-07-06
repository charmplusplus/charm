#include "arrayring.h"

void arrayring_init(void)
{ 
  const int numElements = 50;
  CProxy_arrayRing_array::ckNew(numElements);
}

void arrayring_moduleinit(void){}

arrayRing_array::arrayRing_array()
{
   if(thisIndex==0) {
     CProxy_arrayRing_array carr(thisArrayID);
     carr[0].start(new arrayMessage);
   }
}

void arrayRing_array::start(arrayMessage *msg)
{
  const int maxRings = 10;

  if(!msg->check()) {
    CkAbort("Message corrupted!\n");
  }
  if(thisIndex==0)
    msg->iter++;
  if (msg->iter < maxRings) {
    CProxy_arrayRing_array hr(thisArrayID);
    hr[(thisIndex+1) % ckGetArraySize()].start(msg);
  } else {
    delete msg;
    megatest_finish();
  }
  return;
}

MEGATEST_REGISTER_TEST(arrayring,"fang",1)
#include "arrayring.def.h"
