class main : public CBase_main
{
 public:
  main(CkArgMsg *msg);
  void startBatching();
  void reportInArr();
  void reportInGrp();
  void nextBatch();
  void done();
 private:
  int reportedArr,reportedGrp, completeBatches, arrSize, nBatches, batchSize;
  CProxy_RaceMeArr arrProxy;
};

class RaceMeArr : public CBase_RaceMeArr
{
 public:
 RaceMeArr(int nElements_):nElements(nElements_){}
  void recvMsg();
  RaceMeArr(CkMigrateMessage *m) {};
 private:
  int nElements;
};

class RaceMeGrp : public CBase_RaceMeGrp
{
 public:
  RaceMeGrp(){}
  void recvMsg();
};
