import parallel.PRuntime;
import parallel.RemoteObjectHandle;

public class Ring{
  private int niters;
  private int mype, numpes;
  private RemoteObjectHandle thishandle;
  private Proxy_Ring ring;

  public Ring(RemoteObjectHandle h, InitMsg imsg) {
    thishandle = h;
    niters = imsg.iterations;
    mype = PRuntime.MyPe();
    numpes = PRuntime.NumPes();
    ring = new Proxy_Ring(thishandle);
  }

  public void StartRing(StartMsg s) {
    if(mype==(numpes-1)) {
      if(s.iter == niters) {
        Proxy_Main m = new Proxy_Main(PRuntime.mainhandle);
        ExitMsg emsg = new ExitMsg();
        m.ExitApp(emsg);
      } else {
        s.iter++;
        ring.StartRing(0,s);
      }
    } else {
      ring.StartRing((mype+1)%numpes, s);
    }
  }
}
