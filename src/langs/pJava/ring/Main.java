import parallel.PRuntime;
import parallel.RemoteObject;
import parallel.P;

public class Main extends RemoteObject{
  public static Proxy_RingNode first;

  public static void main(String argv[])
  {
    // new P("Application Started...");
    if (PRuntime.NumPes() < 2)
    {
	  new P("Requires at least two processors!");
	  PRuntime.exit(0);
    }

    RingMessage m = new RingMessage("", null, 0);
    // new P("Creating first ringnode..");
    first = new Proxy_RingNode(0, m);
    // new P("method main returning..");
  }

  public void RingComplete(RingMessage m)
  {
    // new P("Ring Complete");
    Proxy_RingNode last = new Proxy_RingNode(m.sender);
    m.sender = first.thishandle;
    last.SetNext(m);
  }
}
