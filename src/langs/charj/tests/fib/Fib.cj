
package tests;

public mainchare Main {
    public entry Main(CkArgMsg m) {
        int n = atoi(m.argv[1]);
        Fib@ fib = new Fib@(true, n, thishandle);
    }
}

public chare Fib {
    Fib@ parent;
    boolean root;
    int n;
    int partialResult;
    int pendingChildren;

    entry Fib(boolean root_, int n_, Fib@ parent_) {
        CkPrintf("in fib::fib. n=%d\n", n_);
        n = n_;
        root = root_;
        parent = parent_;

        if (n <= 2) {
            partialResult = 1;
            passUp(partialResult);
        }
        partialResult = 0;
        pendingChildren = 2;
        int x = 1;
        Fib@ child1 = new Fib@(false, n-1, thishandle); // replace with "@this"?
        Fib@ child2 = new Fib@(false, n-2, thishandle); // can we get away with just "this"?
    }

    entry void passUp(int subTreeValue) {
        partialResult += subTreeValue;
        if (pendingChildren) return;
        if (root) {
            CkPrintf("Fib(%d) = %d", n, partialResult);
            CkExit();
        }
        parent@passUp(partialResult); // replace with parent@passUp(partialResult_); ?
    }
}
