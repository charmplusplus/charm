package tests.unit;

public class Conditional {
    void test() {
        boolean a=true,b=false,c=false;
        if (true) return;
        if (true)
            return;
        if (a) c=true; else c=false;

        if (b)
            c = true;
        else
            c = false;

        if (c) a = true;
        else b = true;

        if (c) {
            c = true;
        } else {
            c = false;
        }

        c ? a : b || c;

        if (c) {
            a = false;
            b = false;
        } else a = true;
    }
}
