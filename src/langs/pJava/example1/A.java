public class A {
  private RemoteObjectHandle thishandle;

  public A(RemoteObjectHandle h, InitMsg imsg) {
    // do initializations
    thishandle = h;
  }
  public void DoWork(WorkMsg wmsg) {
    // do work
    Proxy_Main m = new Proxy_Main(PRuntime.mainhandle);
    ExitMsg emsg = new ExitMsg();
    m.ExitApp(emsg);
  }
}
