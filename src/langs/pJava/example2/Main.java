import parallel.PRuntime;

public class Main {
  private static double starttime, endtime;
  private static final int NITER = 100;

  public static void main(String argv[]) {
    InitMsg imsg = new InitMsg();
    imsg.iterations = NITER;
    Proxy_Ring ring = new Proxy_ring(imsg);
    StartMsg smsg = new StartMsg();
    smsg.iter = 0;
    ring.StartRing(0, smsg);
    starttime = PRuntime.CTimer();
  }
  public void ExitApp(ExitMsg emsg) {
    endtime = PRuntime.CTimer();
    PRuntime.out.println("Time taken for " + NITER + " rings=" 
                          + (endtime-starttime) + " seconds");
    PRuntime.exit(0);
  }
}
