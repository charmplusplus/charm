
readonly ReductionMain@ mainProxy;

public mainchare ReductionMain {
    public entry ReductionMain(CkArgMsg[~]@ msg) {
        mainProxy = thisProxy;
        int size = 10;
        if (msg.argc < 2) size = atoi(msg.argv[1]);
        Reducer@ arr = new Reducer@(size);
        arr@doWork();
    }

    public entry void done(int result) {
        CkPrintf("All done, result is %d\n", result);
        CkExit();
    }
}

public chare_array [1D] Reducer {
    entry void dowork() {
        contribute(thisIndex,
            CkReduction.sum_int,
            CkCallback(Main.done, mainProxy));
    }
}
