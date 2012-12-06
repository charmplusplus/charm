package tests.unit;

public class Foo {
    int x;
    float y;
}

public mainchare Main {
    public entry Main(CkArgMsg[~]@ m) {
        SimpleChare@ c = new SimpleChare@();
        int x = 0;
        float y = 0.0;
        Foo a = new Foo();
        c@prim1(x);
        c@prim2(x, y);
        c@obj1(a);
        c@obj2(a, a);
        c@mix2(x, a);
        SimpleChare c2;
        c2.local(x, a);
    }
}


public chare SimpleChare {
    int x;
    entry SimpleChare() {}
    entry void prim1(int x) {
    }
    entry void prim2(int x, float y) {
    }
    entry void obj1(Foo a) {
    }
    entry void obj2(Foo a, Foo b) {
    }
    entry void mix2(int x, Foo a) {
    }
    public void local(int x, Foo a) {
    }
}
