import parallel.PRuntime;

public class Main {
  public static void main(String argv[]) {
    InitMsg imsg = new InitMsg();
    Proxy_A a = new Proxy_A(1, imsg);
    WorkMsg wmsg = new WorkMsg();
    a.DoWork(wmsg);
  }
  public void ExitApp(ExitMsg emsg) {
    PRuntime.exit(0);
  }
}
