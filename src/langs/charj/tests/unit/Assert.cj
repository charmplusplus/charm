package tests.unit;

public class Assert {
    void test() {
        int x = 1;
        assert x == 1;
        assert 1 == 1;
        assert 1 == 1: "one equals one";
        assert 1 == 2;
        assert 1 == 2: "one equals two";
    }
}
