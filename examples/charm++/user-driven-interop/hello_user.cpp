#include "hello_user.decl.h"
#include "mpi-interoperate.h"

CProxy_Main main_proxy;
CProxy_Hello hello_array;

class Main : public CBase_Main {
  public:
    Main(CkArgMsg *msg) {
      // Read array size from command line if given
      int array_size = CkNumPes() * 4;
      if (msg->argc > 1) {
        array_size = atoi(msg->argv[1]);
      }

      // Set readonlies and create hello array
      main_proxy = thisProxy;
      hello_array = CProxy_Hello::ckNew(array_size);
    }

    void done() { CkExit(); }
};

class Hello : public CBase_Hello {
  public:
    Hello() {
      CkPrintf("Chare %i created on PE %i\n", thisIndex, CkMyPe());
      num_received = 0;
      contribute(CkCallback(CkReductionTarget(Main, done), main_proxy));
    }
    Hello(CkMigrateMessage* msg) {}

    void sayHello() {
      CkPrintf("Hello from chare %i\n", thisIndex);
      contribute(CkCallback(CkReductionTarget(Main, done), main_proxy));
    }

    void rankReportingIn(int rank) {
      CkPrintf("Chare %i got an ack from %i\n", thisIndex, rank);
      if (++num_received == CkNumPes()) {
        contribute(CkCallback(CkReductionTarget(Main, done), main_proxy));
      }
    }

  private:
    int num_received;
};

// We enter the program with a call to main on every rank (instead of the usual
// charm++ startup with a single main chare as the entry point).
int main(int argc, char** argv) {
  // Initialize the charm runtime, which creates the main chare to do charm
  // specific initialization. Control returns after a CkExit call.
  CharmInit(argc, argv);

  CkPrintf("Starting in user driven mode on %i\n", CkMyPe());

  // Send a single broadcast to the chare array from PE 0. These messages won't
  // be sent/received until the charm scheduler is started.
  if (CkMyPe() == 0) {
    hello_array.sayHello();
  }

  /** 
    * Do other user code work here as needed, can include MPI code
    **/

  // Start the charm scheduler. Doesn't return until CkExit() is called.
  StartCharmScheduler();

  // Send some more messages, this time from every rank. As before these won't
  // be sent until the scheduler starts running.
  hello_array.rankReportingIn(CkMyPe());

  // Start the charm scheduler again so the messages can be sent/received
  StartCharmScheduler();

  CharmLibExit();
  return 0;
}

#include "hello_user.def.h"
