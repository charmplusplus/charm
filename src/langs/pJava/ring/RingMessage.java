import parallel.RemoteObjectHandle;
import parallel.Message;

class RingMessage extends Message{
  public String text;
  public RemoteObjectHandle sender;
  public int iter;
  
  RingMessage(String txt, RemoteObjectHandle h, int i)
  {
    text = txt;
    sender = h;
    iter = i;
  }
}
