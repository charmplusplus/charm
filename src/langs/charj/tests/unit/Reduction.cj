
readonly ReductionMain@ mainProxy;
readonly int units;
readonly double d;

public mainchare ReductionMain {
      public entry ReductionMain(CkArgMsg msg) {
          mainProxy = thisProxy;
          if (msg.argc < 2) CkExit();
          units = atoi(msg.argv[1]);
          d = atof(msg.argv[2]);
      }

      public entry void reportIn(CkReductionMsg msg) {
      }

      public entry void done() {
          CkPrintf("Add done\n");
          CkExit();
      }
}

public chare_array [1D] Reducer {
    private float myfloat;

    void dowork() {
        double out = d + (double)thisIndex;
        contribute(2*sizeof(double), out, CkReduction::sum_double);
    }
}
