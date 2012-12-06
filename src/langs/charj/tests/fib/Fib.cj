
readonly Main@ mainProxy;

extern atoi;

public mainchare Main {
    int n;

    public entry Main(CkArgMsg[~]@ m) {
        if (m.argc < 2) n = 16;
        else n = atoi(m.argv[1]);
        mainProxy = thisProxy;
        Fib@ fib = new Fib@(true, n, thishandle);
    }

    public entry void done(int value) {
        CkPrintf("Fib(%d) = %d\n", n, value);
        CkExit();
    }
}

public chare Fib {
    Fib@ parent;
    boolean root;
    int n;
    int partialResult;
    int pendingChildren;

    private int seq_fib(int n) {
        if (n < 2) return n;
        return seq_fib(n-1) + seq_fib(n-2);
    }

    public entry Fib(boolean root_, int n_, Fib@ parent_) {
        n = n_;
        root = root_;
        parent = parent_;

        if (n <= 10) {
            partialResult = seq_fib(n);
            passUp();
        } else {
            Fib@ child1 = new Fib@(false, n-1, thisProxy);
            Fib@ child2 = new Fib@(false, n-2, thisProxy);
            partialResult = 0;
            pendingChildren = 2;
        }
    }

    public traced entry void gather(int value) {
        partialResult += value;
        if (--pendingChildren == 0) passUp();
    }

    public traced void passUp() {
        if (root) mainProxy@done(partialResult);
        else parent@gather(partialResult);
        delete this;
    }
}
