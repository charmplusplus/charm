import parallel.PRuntime;
import parallel.RemoteObject;
import parallel.P;

public class Main extends RemoteObject{
  public static Proxy_HelloNode group;
  private static int nExited;

  public static void main(String argv[])
  {
    // new P("Application Started...");

    nExited = 0;
    HelloMessage m = new HelloMessage();
    // new P("Creating first ringnode..");
    group = new Proxy_HelloNode(m);
    // new P("method main returning..");
  }

  public void Exit(HelloMessage m)
  {
    // new P("Ring Complete");
    nExited++;
    if(nExited==PRuntime.NumPes())
      PRuntime.exit(0);
  }
}
