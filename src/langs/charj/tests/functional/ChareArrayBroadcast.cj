package tests.unit;

public chare_array [1d] SimpleChareArray {
    int x;
    public entry SimpleChareArray() {}
    public entry void test() {
        CkPrintf("Testing\n");
    }
}

public mainchare Main {
    public entry Main(CkArgMsg[~]@ m) {
        SimpleChareArray@ a = new SimpleChareArray@(10);
        a@test();
    }
}
