package tests.unit;

public class WhileLoop {
    void test() {
        boolean a;
        while (a) {
            a = !a;
        }
        while (!a) a=!a;
        while(a) break;
    }
}
