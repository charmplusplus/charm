import parallel.PRuntime;
import parallel.RemoteObjectHandle;

public class Proxy_Test {
  public static int classID;

  private static int Entry1_Message1;
  private static int Entry2_Message2;
  private static int Test_Message0;
  private static int Entry3_Message3;

  private static int Message1_ID;
  private static int Message2_ID;
  private static int Message0_ID;
  private static int Message3_ID;

  private RemoteObjectHandle thishandle;

  public Proxy_Test(RemoteObjectHandle handle) {
    thishandle = handle;
  }

  public void Entry1(Message1 m) {
    PRuntime.InvokeMethod(thishandle,Entry1_Message1,m);
  }

  public void Entry2(Message2 m) {
    PRuntime.InvokeMethod(thishandle,Entry2_Message2,m);
  }

  public Proxy_Test(int pe, Message0 m) {
    thishandle = PRuntime.CreateRemoteObject(pe, classID,
      Test_Message0, m);
  }

  public void Entry3(Message3 m) {
    PRuntime.InvokeMethod(thishandle,Entry3_Message3,m);
  }

  static {
    classID = PRuntime.RegisterClass("Test");
    Message1_ID = PRuntime.GetMessageID("Message1");
    Message2_ID = PRuntime.GetMessageID("Message2");
    Message0_ID = PRuntime.GetMessageID("Message0");
    Message3_ID = PRuntime.GetMessageID("Message3");

    Entry1_Message1 = PRuntime.RegisterEntry("Entry1", classID, Message1_ID);
    Entry2_Message2 = PRuntime.RegisterEntry("Entry2", classID, Message2_ID);
    Test_Message0 = PRuntime.RegisterConstructor(classID, Message0_ID);
    Entry3_Message3 = PRuntime.RegisterEntry("Entry3", classID, Message3_ID);
  }
}
