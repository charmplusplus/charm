class LBUserDataStruct {
  public:
  int idx;
  LDObjHandle handle;

  void pup(PUP::er &p) {
    p|idx;
    p|handle;
  }
};
