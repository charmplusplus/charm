
package tests;

readonly Main@ main;

public mainchare Main {
    int n;

    public entry Main(CkArgMsg m) {
        n = atoi(m.argv[1]);
        main = thishandle;
        Fib@ fib = new Fib@(true, n, NULL);
    }

    public entry void done(int value) {
        CkPrintf("Fib(%d) = %d", n, value);
        CkExit();
    }
}

public chare Fib {
    Fib@ parent;
    boolean root;
    int n;
    int partialResult;
    int pendingChildren;

    public entry Fib(boolean root_, int n_, Fib@ parent_) {
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
        Fib@ child2 = new Fib@(false, n-2, thishandle);
    }

    public entry void passUp(int subTreeValue) {
        partialResult += subTreeValue;
        if (pendingChildren) return;
        if (root) main@done(partialResult);
        else parent@passUp(partialResult);
    }
}
