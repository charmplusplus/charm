package tests.unit;

public class ForLoop {
    void test() {
        int j;
        for (int i=0; i<1; ++i) {
            ++i;
        }
        for (int i=0; ; ) { break; }
        for (;j!=0;) {}
        for (;;j++) {}
        for (;;) break;
        for (int i=0;;++i) {}
        for (int i=0, j=1;;++i, ++j) break;
        for (j=0; j<10; ++j) break;
    }
}
