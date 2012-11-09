
readonly Main@ mainProxy;

public mainchare Main {
    ArrayTest@ array;

    public entry Main(CkArgMsg[~]@ m) {
        mainProxy = thisProxy;
        array = new ArrayTest@(10);
        start();
    }

    public entry void start() {
        array@start();
    }

    public entry void finished() {
        CkExit();
    }
}

public chare_array [1d] ArrayTest {
    public entry ArrayTest() { }

    public entry void start() {
        contribute(CkCallback(CkReductionTarget(Main, finished), mainProxy));
    }
}
