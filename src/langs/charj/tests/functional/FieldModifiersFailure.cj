// expect failure
package tests.unit;

public class FieldModifiers {
    int z;
    public int i;
    private int j;
    static int k;
    public static int l;
    volatile int m;
    protected int n;
    final int r;
// should fail
    entry int o; 
    abstract int p; 
    transient int q;
}
