#ifdef HELLO_USE_MAINCHARE
#include "hello_user_with_main.decl.h"
#else
#include "hello_user.decl.h"
#endif
#include "mpi-interoperate.h"

CProxy_Hello helloProxy;

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
class Main : public CBase_Main {
  public:
    Main(CkArgMsg *msg) {
      // Call the init function using the message argc and argv
      init_hello(msg->argc, msg->argv);
      delete msg;
    }
};
#endif

class Hello : public CBase_Hello {
  public:
    Hello() : num_received(0) {
      CkPrintf("Chare %i created on PE %i\n", thisIndex, CkMyPe());
    }

    Hello(CkMigrateMessage* msg) {}

    void sayHello() {
      CkPrintf("Hello from chare %i\n", thisIndex);
      contribute(CkCallback(CkCallback::ckExit));
    }

    void rankReportingIn(int rank) {
      CkPrintf("Chare %i got an ack from %i\n", thisIndex, rank);
      if (++num_received == CkNumNodes()) {
        contribute(CkCallback(CkCallback::ckExit));
      }
    }

  private:
    int num_received;
};

// We enter the program with a call to main on every rank (instead of the usual
// charm++ startup with a single main chare as the entry point).
int main(int argc, char** argv) {
// If we are not using a mainchare, and want to still use readonly variables
// we can do a split initialization.
#ifndef HELLO_USE_MAINCHARE
  // Start CharmInit, but don't complete it until after we've set readonlies.
  CharmBeginInit(argc, argv);

  // Initialize readonlies and create any initial groups/nodegroups from PE 0.
  if (CkMyPe() == 0) {
    init_hello(argc, argv);
  }

  // Tell the runtime to finish initialization, which sends out readonlies and
  // any other initialization messages. Returns when initialization is done.
  CharmFinishInit();
#else
  // If we have a mainchare, or do not use readonlies, we can let CharmInit do
  // the full initialization in one shot.
  CharmInit(argc, argv);
#endif
  CkPrintf("Starting in user driven mode on %i\n", CkMyPe());

  // Broadcast to the chare array from PE 0. Nothing is sent until we start the
  // Charm scheduler.
  if (CkMyPe() == 0) {
    helloProxy.sayHello();
  }

  ////
  // Do other user code work here as needed
  ////

  // Start the charm scheduler on all ranks. Returns after CkExit() is called.
  StartCharmScheduler();

  // Send some more messages, this time from every rank. As before these won't
  // be sent until the scheduler starts running.
  helloProxy.rankReportingIn(CkMyPe());

  // Start the charm scheduler again so the messages can be sent/received.
  StartCharmScheduler();

  // Cleans up runtime and exits the program.
  CharmLibExit();
  return 0;
}

#ifdef HELLO_USE_MAINCHARE
#include "hello_user_with_main.def.h"
#else
#include "hello_user.def.h"
#endif
