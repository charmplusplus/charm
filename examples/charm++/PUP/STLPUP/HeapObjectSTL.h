#include <vector>
#include <pup_stl.h>
class HeapObject
{

 public:

  int publicInt;

 HeapObject(int param1, bool param2):publicInt(param1), privateBool(param2) {}

 HeapObject():publicInt(0), privateBool(false) {}
 void pup(PUP::er &p)
 {
   // remember to pup your superclass if there is one
   p|publicInt;
   p|privateBool;
   p|data;
 }
 inline HeapObject &operator=(const HeapObject &indata) {
    publicInt=indata.publicInt;
    privateBool=indata.privateBool;
    data=indata.data;
    return *this;
  }


 void doWork()
 {
   // here is where a useful object would do something
   privateBool=publicInt<20;
 }

 ~HeapObject(){}

 private:

  // PUP is orthogonal to public vs private member choices
  bool privateBool;
  std::vector<float> data;
};
