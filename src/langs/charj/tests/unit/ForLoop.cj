package tests.unit;

public class ForLoop {
    void test() {
        for (int i=0; i<1; ++i) {
            ++i;
        }
        for (;;) {}
        for (int i=0;;++i) break;
        for (int i=0, j=1;;++i, ++j) break;
    }
}
