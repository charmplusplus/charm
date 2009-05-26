package tests.unit;

public class ForLoop {
    void test() {
        for (int i=0; i<1; ++i) {
            ++i;
        }
        for (;;) {}
        for (int i=0;;++i) break;
    }
}
