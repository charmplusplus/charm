package tests.unit;

public class Foo {
}

public class FuncArgs {
    void test() {
        Foo f;
        a1(f);
        a2(a1(f), 2);
    }
    int a1(Foo x) { return 0; }
    void a2(int x, float y) {}
}
