import parallel.PRuntime;
import parallel.RemoteObject;
import parallel.P;

public class HelloNode extends RemoteObject{
  Proxy_Main main;

  public HelloNode(HelloMessage m)
  {
    super(m.handle);
    new P("Hello World !!");
    main = new Proxy_Main(PRuntime.mainhandle);
    main.Exit(m);
  }
}
