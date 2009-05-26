package tests.unit;

public class Switch {
    void test() {
        int a;
        switch (a) {
            case 1:
                break;
            case 2:
            case 3:
            default:
                a=0;
        }
    }
}
