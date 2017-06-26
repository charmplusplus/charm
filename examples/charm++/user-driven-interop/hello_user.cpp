#include "hello_user.decl.h"
#include "mpi-interoperate.h"

CProxy_Hello helloProxy;

// #define HELLO_USE_MAINCHARE

void init_hello(int argc, char** argv) {
  int array_size = CkNumPes() * 4;

  // Read array size from command line if given
  if (argc > 1) {
    array_size = atoi(argv[1]);
  }

  CkAssert(array_size >= CkNumPes() && "This program requires at least as many array elements as the number of PEs.");

  helloProxy = CProxy_Hello::ckNew(array_size);
}

#ifdef HELLO_USE_MAINCHARE
CProxy_Main mainProxy;
class Main : public CBase_Main {
  public:
    Main(CkArgMsg *msg) {
      // Call the init function using the message argc and argv
      init_hello(msg->argc, msg->argv);
      // Set readonlies and create hello array
      mainProxy = thisProxy;
    }

    void done() {
      CkExit();
    }
};
#endif

class Hello : public CBase_Hello {
  public:
    Hello() {
      CkPrintf("Chare %i created on PE %i\n", thisIndex, CkMyPe());
      num_received = 0;

      int numPes = CkNumPes();
      received   = new bool[numPes];

      for (int i = 0; i < numPes; i++) {
        received[i] = false;
      }
    }

    Hello(CkMigrateMessage* msg) {}

    void sayHello() {
      CkPrintf("Hello from chare %i\n", thisIndex);
#ifdef HELLO_USE_MAINCHARE
        contribute(CkCallback(CkReductionTarget(Main, done), mainProxy));
#else
        contribute(CkCallback(CkReductionTarget(Hello, done), thisProxy[0]));
#endif
    }

    void rankReportingIn(int rank) {
      if (received[rank]) {
        return;
      } else if (++num_received == CkNumPes()) {
#ifdef HELLO_USE_MAINCHARE
        contribute(CkCallback(CkReductionTarget(Main, done), mainProxy));
#else
        contribute(CkCallback(CkReductionTarget(Hello, done), thisProxy[0]));
#endif
      } else if (rank != CkMyPe() && !received[CkMyPe()]) {
        thisProxy.rankReportingIn(CkMyPe());
      }

      CkPrintf("Chare %i got an ack from %i\n", thisIndex, rank);
      received[rank] = true;
    }

    void done() {
      CkExit();
    }

  private:
    int num_received;
    bool *received;
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
#ifndef HELLO_USE_MAINCHARE
    init_hello(argc, argv);
#endif

    helloProxy.sayHello();
  }

  /**
    * Do other user code work here as needed, can include MPI code
    **/

  // Start the charm scheduler. Doesn't return until CkExit() is called.
  StartCharmScheduler();

  if (CkMyPe() == 0) {
    // Send some more messages, this time from every rank. As before these won't
    // be sent until the scheduler starts running.
    helloProxy.rankReportingIn(CkMyPe());
  }

  // Start the charm scheduler again so the messages can be sent/received
  StartCharmScheduler();

  CharmLibExit();
  return 0;
}

#include "hello_user.def.h"
