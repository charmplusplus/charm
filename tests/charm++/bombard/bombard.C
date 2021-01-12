#include "bombard.decl.h"

CProxy_main mProxy;

class main : public CBase_main {
    int receiver;
    int sender;
    int count;
    int arraySize;
    int mode;
    int counter;

  public:
    main(CkArgMsg *msg) {

      if(msg->argc != 4) {
        CmiPrintf(" Need 4 arguments: Usage ./bombard <mode> <arraySize> <count>\n");
        CkExit();
      }

      mode = atoi(msg->argv[1]);
      arraySize = atoi(msg->argv[2]);
      count = atoi(msg->argv[3]);

      counter = 0;
      mProxy = thisProxy;

      if(mode < 0 || mode > 2) {
        CkAbort("Mode should be 1 (p2p), 2 (all to one) or 3 (all to all)\n");
      }

      CmiPrintf("================================================================================\n");
      CmiPrintf("[%d][%d][%d] Main Chare - Begin Bombard: mode:%d, arraySize:%d, count:%d, num_pes:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), mode, arraySize, count, CkNumPes());
      CmiPrintf("================================================================================\n");

      CProxy_grp gProxy = CProxy_grp::ckNew(mode, arraySize, count);
    }

    void done() {
      CmiPrintf("================================================================================\n");
      CmiPrintf("[%d][%d][%d] Main Chare - Program Complete: mode:%d, arraySize:%d, count:%d, num_pes:%d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), mode, arraySize, count, CkNumPes());
      CmiPrintf("================================================================================\n");
      CkExit();
    }
};

class grp : public CBase_grp {

  char *buffer1, *buffer2;
  int sendCount;
  int recvCount;
  int arraySize;
  int counter;
  int mode;

  CkCallback reductionCb;

  public:
    void bombard() {

      if(mode == 0) {
        // p2p case - first PE sends to last PE
        if(CkMyPe() == 0) {
          for(int i = 0; i < sendCount; i++) {
            if(arraySize > 0)
              thisProxy[CkNumPes()-1].receive(buffer1, arraySize);
            else
              thisProxy[CkNumPes()-1].receive(arraySize);
          }
        }

        if(CkMyPe() != CkNumPes() - 1) {
          // contribute to reduction
          contribute(reductionCb);
        }

      } else if(mode == 1) {
        // all to one - all PEs send to last PE

        if(CkMyPe() != CkNumPes() - 1 || CkMyPe() == 0) {
          for(int i = 0; i < sendCount; i++) {
            if(arraySize > 0)
              thisProxy[CkNumPes()-1].receive(buffer1, arraySize);
            else
              thisProxy[CkNumPes()-1].receive(arraySize);
          }
        }

        if(CkMyPe() != CkNumPes() - 1) {
          // contribute to reduction
          contribute(reductionCb);
        }

      } else if(mode == 2) {
        // all to all - all PEs send to all PEs

        for(int i = 0; i < sendCount; i++) {
          if(arraySize > 0)
            thisProxy.receive(buffer1, arraySize);
          else
            thisProxy.receive(arraySize);
        }
      } else {
        CkAbort("bombard: Invalid Mode\n");
      }
    }

    void progress() {

      if(mode == 0) {

        CkAssert(CkMyPe() == CkNumPes() - 1);
        CmiPrintf("[%d][%d][%d] p2p Receiver: ITER = %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), counter);

        recvCount++;

        if(recvCount == sendCount) {
          CmiPrintf("[%d][%d][%d] p2p Receiver completed %d iterations\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), recvCount);
          // contribute to reduction
          contribute(reductionCb);
        }

      } else if(mode == 1 ) {

        CkAssert(CkMyPe() == CkNumPes() - 1);

        if((counter == CkNumPes() - 1) || (CkNumPes() == 1 && counter == 1)) {
          counter = 0;
          recvCount++;
          CmiPrintf("[%d][%d][%d] allToOne Receiver: ITER = %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), recvCount);

          if(recvCount == sendCount) {
            CmiPrintf("[%d][%d][%d] allToOne Receiver completed %d iterations\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), recvCount);
            // contribute to reduction
            contribute(reductionCb);
          }
        }

      } else if(mode == 2) {

        if(counter == CkNumPes()) {
          CmiPrintf("[%d][%d][%d] allToall Receiver: ITER = %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), recvCount);
          counter = 0;
          recvCount++;

          if(recvCount == sendCount) {
            CmiPrintf("[%d][%d][%d] allToOne Receiver completed %d iterations\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), recvCount);

            // contribute to reduction
            contribute(reductionCb);
          }
        }
      } else {
        CkAbort("progress: Invalid Mode\n");
      }
    }

    // contructor
    grp(int mode, int arraySize, int count) {

      this->mode = mode;
      this->arraySize = arraySize;
      this->sendCount = count;
      this->recvCount = 0;

      if(arraySize > 0) {
        buffer1 =  new char[arraySize];
      } else {
        buffer1 = NULL;
      }
      counter = 0;

      reductionCb = CkCallback(CkIndex_main::done(), mProxy);

      CmiPrintf("[%d][%d][%d] Group 'grp' constructor\n", CmiMyPe(), CmiMyNode(), CmiMyRank());
      bombard();
    }

    void receive() {
      counter++;
      progress();
    }

    void receive(int size) {
      counter++;
      progress();
    }

    void receive(char *buffer, int size) {
      counter++;
      progress();
    }
};

#include "bombard.def.h"
