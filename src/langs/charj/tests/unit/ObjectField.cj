package tests.unit;

public class Foo {
    //public Foo(int _x) { property = _x; }
    public int property;
}

public class Bar {
    public int property;
    public Foo foo;
}

public class SimpleClass {
    Bar m;

    void set_m(Bar x) {
        m = x;
        m.property++;
        m.property = 0;
        m.foo.property++;
        method2(m.property, m.foo);
        //Foo n = new Foo(m.foo.property);
        //this.method(m.property);
        //method(n.property);
        //delete n;
    }

    void method1(int x) { }
    void method2(int x, Foo f) { }

    Bar get_m() {
        return m;
    }

    void test() {
    }
}
