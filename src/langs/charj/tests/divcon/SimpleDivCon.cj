
package tests;

readonly Main@ main;

public mainchare Main {
    public entry Main(CkArgMsg m) { }
}

// Simple example of divcon use, directly translated from TMS example
// (https://charm.cs.uiuc.edu/private/tms/listlog.php?param=834)
public chare DivCon {
    private int M;

    public entry DivCon() {
        M = 10;
    }

    private int f(int x) { return x; }
    private int g(int x, int y) { return x; }

    private int a(int x) { return x; }
    private int b(int x, int y) { return x; }
    private int c(int x) { return x; }
    private int d(int x) { return x; }

    public int f(int n) {
        if (n < M) {
            let
                x1 = a(n),
                x2 = b(x1, n)
            in
                f(x1 + x2);
        } else {
            let
                x1 = c(n),
                x2 = d(n)
            in
                g(x1, x2);
        }
    }
}
