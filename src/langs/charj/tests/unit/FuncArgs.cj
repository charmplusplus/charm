package tests.unit;

public class FuncArgs {
    void test() {
        a1(1);
        a2(a1(2), 2);
    }
    int a1(MadeUpType x) {}
    void a2(int x, float y) {}
}
