package tests.Hello;
import charj.lang.someObject;

public class Hello extends Chare {
    public Hello() {}
    public void sayHello(int payload) {
        MyObject someObject = new MyObject(1, 2, "hello", new Array<int>(1));
        int[] myArray = new int[25];
        Array<int> myGenericArray = new Array<int>(25);
        for (int i=0; i<25; ++i) {
            myArray[i] = payload + i;
            myGenericArray[i] = payload * i;
        }
    }
}

public interface INT extends I1, I2 {
}
