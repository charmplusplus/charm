package tests.Hello;
//import charj.lang.someObject;

public class Hello {
    entry Hello() {}
    public Hello(int x) {}
    private Hello(int x, int y) {}
    entry void sayHello(int payload) {
        int y = 1;
        embed "c++" {{
            int* x = &y;
        }}
        Hello h = new Hello();
        //MyObject someObject = new MyObject(1, 2, "hello", new Array<int>(1));
        //int myArray[25] = new int[25];
        //Array<int> myGenericArray = new Array<int>(25);
        //for (int i=0; i<25; ++i) {
        //    myArray[i] = payload + i;
        //    myGenericArray[i] = payload * i;
        //}
    }
    // Non-void entry methods aren't handled yet.
    //entry int SayHello() { }
    public void anotherMethod() { }
    public int anotherMethod(int x) { return x; }
    private void aVoidMethod() {}
    private int aNonVoidMethod(int x) { return x; }
}

