class SimpleObject
{

 public:

  int publicInt;

 SimpleObject(int param1, bool param2):publicInt(param1), privateBool(param2) {}

 SimpleObject():publicInt(0), privateBool(false) {}
 void pup(PUP::er &p)
 {
      // remember to pup your superclass if there is one
   p|publicInt;
   p|privateBool;
 }

 void doWork()
 {
   // here is where a useful object would do something
   publicInt++;
   privateBool=publicInt<20;
 }

 ~SimpleObject(){}

 private:

  // PUP is orthogonal to public vs private member choices
  bool privateBool;

};
