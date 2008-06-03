public class Hello extends Chare {
    public Hello() {}
    public void sayHello(int payload) {
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
