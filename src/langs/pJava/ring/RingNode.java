import parallel.PRuntime;
import parallel.RemoteObjectHandle;
import parallel.RemoteObject;
import parallel.P;

public class RingNode extends RemoteObject{
  int mype, numpes;
  Proxy_RingNode prev; 
  Proxy_RingNode next; 
  Proxy_Main main;
  static final int MAXITER = 100;
  double starttime;

  public RingNode(RingMessage m)
  {
    super(m.handle);
    // new P("Creating RingNode..");
    mype = PRuntime.MyPe();
    numpes = PRuntime.NumPes();
    // new P(thishandle.toString());
    if(m.sender == null) {
      // new P("sender is null");
      m.sender = (RemoteObjectHandle) thishandle.clone();
      next = new Proxy_RingNode((mype+1)%numpes, m);
      // new P(next.thishandle.toString());
    } else {
      // new P(m.sender.toString());
      prev = new Proxy_RingNode(m.sender);
      if(mype != numpes-1) {
        m.sender = (RemoteObjectHandle) thishandle.clone();
        next = new Proxy_RingNode((mype+1)%numpes, m);
      } else {
        main = new Proxy_Main(PRuntime.mainhandle);
        m.sender = (RemoteObjectHandle) thishandle.clone();
        main.RingComplete(m);
      }
    }
  }

  public void SetNext(RingMessage m)
  {
    // new P("SetNext Called");
    next = new Proxy_RingNode(m.sender);
    m.sender = (RemoteObjectHandle) thishandle.clone();
    m.text = new String("Hello from processor " + mype + "!!");
    starttime = PRuntime.CTimer();
    next.StartRing(m);
  }

  public void StartRing(RingMessage m)
  {
    // new P("StartRing called");
    m.iter = 0;
    // new P("On " + mype + ": " + m.text + ", Iter=" + m.iter);
    prev = new Proxy_RingNode(m.sender);
    m.text = new String("Hello from processor " + mype + "!!");
    next.Forward(m);
  }

  public void Forward(RingMessage m)
  {
    // new P("On " + mype + ": " + m.text + ", Iter=" + m.iter);
    if(m.iter == MAXITER && mype==(numpes-1)) {
      double timeTaken = PRuntime.CTimer() - starttime;
      new P("Ring completed in " + timeTaken + " seconds.");
      PRuntime.exit(0);
    } else {
      if(mype==0)
        m.iter++;
	  m.text = new String("Hello from processor " + mype + "!!");
	  next.Forward(m);
    }
  }
}
