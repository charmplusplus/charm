class HeapObject {
 public:
  int publicInt;

  HeapObject(int param1, bool param2) : publicInt(param1), privateBool(param2) {
    data = new float[publicInt];
  }

  HeapObject() : publicInt(0), privateBool(false) {
    data = NULL;
  }

  ~HeapObject() {
    if (publicInt > 0)
      delete [] data;
  }

  void pup(PUP::er &p) {
    // remember to pup your superclass if there is one
    p|publicInt;
    p|privateBool;
    if (p.isUnpacking())
      data = new float[publicInt];
    PUParray(p, data, publicInt);
  }

  inline HeapObject &operator=(const HeapObject &indata) {
    if (data && publicInt > 0)
      delete [] data;
    publicInt = indata.publicInt;
    privateBool = indata.privateBool;
    if (publicInt > 0)
      data = new float[publicInt];
    for (int i = 0; i < publicInt; ++i)
      data[i] = indata.data[i];
    return *this;
  }

  void doWork() {
    // here is where a useful object would do something
    privateBool = publicInt < 20;
  }

 private:
  // PUP is orthogonal to public vs private member choices
  bool privateBool;
  float *data;
};
