import parallel.PRuntime;
import java.lang.Class;

public class RegisterAll {

  static void registerAll() {
    PRuntime.RegisterMessage("Message0");
    PRuntime.RegisterMessage("Message1");
    PRuntime.RegisterMessage("Message2");
    PRuntime.RegisterMessage("Message3");
    Class.FindClass("Proxy_Test");
  }
}
