/*
Java Interface file for Converse Client/Server 
from the Parallel Programming Lab, Univ. of Illinois at Urbana-Champaign

Converse is the runtime support system for Charm++,
a parallel programming language developed at the UIUC PPL.
You can read about Charm++ at
	http://charm.cs.uiuc.edu/
	
Orion Sky Lawlor, olawlor@acm.org, 10/15/2000
*/
import java.net.*;
import java.io.*;
import java.util.*;
import java.security.MessageDigest;

/** 
 * This class is used to talk back and forth with a Converse 
 * parallel program.
 * It can sent requests to and receive responses from the program.
 * 
 * @author  
 * <a href="mailto:olawlor@acm.org">Orion Lawlor</a>, UIUC 
 * <a href="http://charm.cs.uiuc.edu/">Parallel Programming Lab</a>
 * @version 2.0
 */

public class CcsServer 
{ 
// --------------------------------------------------------------------------
   /** Establish a connection with a running Converse program.
    *
    * @param address gives the IP address of the running program.
    * @param port gives the TCP port number of the program's server socket.
    */
   public CcsServer(InetAddress address, int port,byte[] secretKey)
   	throws IOException
   {
   	connect(address,port,secretKey);
   }
   /** Establish a connection with a running Converse program.
    *
    * @param host gives the name of the machine of the running program.
    * @param port gives the TCP port number of the program's server socket.
    */
   public CcsServer(String host,int port,byte[] secretKey)
   	throws IOException, UnknownHostException
   {
   	InetAddress ip=InetAddress.getByName(host);
   	connect(ip,port,secretKey);
   }
   /** Establish a connection with a running Converse program.
    *
    * @param init gives the CCS initialization string, printed by converse
    *  during startup.
    */
   public CcsServer(String init,byte[] secretKey)
   	throws IOException, NumberFormatException
   {
   //Format of the init. string is:
   // "ccs: Server IP = "+int(ip)+", Server port = "+port+" $"
   	int firstEq=init.indexOf("=");
   	int secondEq=init.indexOf("=",firstEq+1);
   	String ip_str=init.substring(firstEq+1,init.indexOf(",")).trim();
   	String port_str=init.substring(secondEq+1,init.indexOf("$")).trim();
   	
   	//Convert string ip and port to numeric IP and port
   	long ip_int=Long.parseLong(ip_str);
   	int port=Integer.parseInt(port_str);
   	
   	//Convert the IP integer to a dotted decimal
   	if (ip_int<0) ip_int+=(1L<<32); //Make unsigned
   	String ip_dot="";
   	for (int b=3;b>=0;b--) {
   		Long l=new Long(0xff&(ip_int>>(8*b)));
		ip_dot=ip_dot+l.toString();
		if (b>0) ip_dot=ip_dot+".";
	}
   	
   	//Convert the dotted decimal IP to a InetAddr
	InetAddress ip=null;
   	try {
   		ip=InetAddress.getByName(ip_dot);
   	}
   	catch (UnknownHostException e)
   	  { throw new NumberFormatException(e.toString()); }
   	  
   	//Finally, connect to the server
   	connect(ip,port,secretKey);
   }

//-------------------- Send Requests/Recv Responses -----------------
    public static class Request {
	protected Socket sock;
	protected int salt;
	protected Request(Socket sock_,int salt_) {
	    sock=sock_;
	    salt=salt_;
	}
    };
    private Request lastRequest;

   /** Send a request to a Converse program.  This executes a CCS "handler"
    * (registered in the parallel program with CcsRegisterHandler)
    * on the given processor.
    *
    * @param handlerName gives the name of the CCS handler to process the request.
    * @param destPe gives the (0-based) processor number to process the request.
    */
   public Request sendRequest(String handlerName, int destPe)
   	throws IOException {return sendRequest(handlerName,destPe,null);}
   /** Send a request to a Converse program.  This executes a CCS "handler"
    * (registered in the parallel program with CcsRegisterHandler)
    * on the given processor with the given data.
    *
    * @param handlerName gives the name of the CCS handler to process the request.
    * @param destPe gives the (0-based) processor number to process the request.
    * @param data gives the data to pass to the handler, if any.
    */
   public Request sendRequest(String handlerName, int destPe, byte []data)
   	throws IOException
   {
   	//Open a socket and send the request header
	debug("  Connecting for request '"+handlerName+"'");
   	Socket sock=new Socket(hostIP,hostPort);
	debug("  Connected.  Sending header");
   	DataOutputStream o=new DataOutputStream(sock.getOutputStream());
	int dataLen=0;
	if (data!=null) dataLen=data.length;

   	//Create an outgoing message header
	int handlerOff=8, handlerMAX=32;
	int headerLen=handlerOff+handlerMAX;
	byte[] header=new byte[headerLen];
	writeInt(header,0,dataLen);
	writeInt(header,4,destPe);
	writeString(header,8,handlerMAX,handlerName);

	int salt=0;
	if (isAuth) { /*Send authentication header*/
	    debug("  Sending authentication for level "+level);
	    o.writeInt(0x80000000|level); /*Request type: ordinary message*/
	    salt=rand.nextInt();
	    o.writeInt(clientID);
	    o.writeInt(salt);
	    o.write(SHA_makeHash(key,clientSalt++,header));
	}

   	o.write(header,0,headerLen);
   	debug("  Header sent.  Sending "+dataLen+" bytes of request data");

   	//Send any associated data
   	if (data!=null)
   		o.write(data);
   	o.flush();
   	debug("  Request sent");
	
   	//socket is left open for reply
        Request r=new Request(sock,salt);
	lastRequest=r;
   	return r;
   }
   /** Wait for a response from Converse program.  This refers to the last
    * executed request, and will wait indefinitely for the response.
    * The response data will be returned as a byte array.
    */
   public byte[] recvResponse() throws IOException
   {return recvResponse(lastRequest);}

   /** Wait for a response from Converse program.  This uses the returned value from any
    * previous sendRequest, and will wait indefinitely for the response.
    * The response data will be returned as a byte array.
    */
   public byte[] recvResponse(Request r) throws IOException
   {
	debug("  Waiting for response");
   	DataInputStream i=new DataInputStream(r.sock.getInputStream());
	if (isAuth) {
	    byte[] hash=new byte[SHA_len];
	    i.readFully(hash);
	    if (!SHA_checkHash(key,r.salt,null,hash)) 
		abort("Server's key does not match ours (during response)!");
	}
   	int replyLen=i.readInt(); 
	debug("  Response will be "+replyLen+" bytes");	
   	byte[] reply=new byte[replyLen];
        i.readFully(reply);  //All data may not come in the same packet.
   	//if (replyLen!=i.read(reply,0,replyLen))
   	//	throw new IOException("CCS Reply socket closed early!");
	debug("  Got entire response");
   	r.sock.close();
	r.sock=null;
   	return reply;
   }
   
   /** Determine if a response is pending on the given socket
     */
   public boolean hasResponse(Request r) throws IOException
   {
   	if (r.sock.getInputStream().available()>0)
   		return true;
   	else
   		return false;
   }
      
   /** Close given CCS request socket.
    */
   public void close(Request r)
   {
   	if (r.sock!=null) {
   		try {
   			r.sock.close();
		}
		catch (Exception e) { /*ignore*/ }
   		r.sock=null;
   	}
   }
   /** Close current CCS request socket. */
   public void close() {close(lastRequest);lastRequest=null;}

//----------------- Program Info -------------
   /** Get the number of nodes (address spaces) in parallel machine.*/
   public int getNumNodes() {return numNodes;}
   /** Get the number of processors in parallel machine.*/
   public int getNumPes() {return numPes;}
   /** Get the (0-based) number of the first processor on the given (0-based) node.*/
   public int getNodeFirst(int node) {return nodeFirst[node];}
   /** Get the total number of processors on the given (0-based) node.*/
   public int getNodeSize(int node) {return nodeSize[node];}


    static final public int readInt(byte[] dest,int destStart)
    {
	return ((0xff&dest[destStart+0])<<24)+
	       ((0xff&dest[destStart+1])<<16)+
	       ((0xff&dest[destStart+2])<< 8)+
	       ((0xff&dest[destStart+3])<< 0);
    }
    static final public void writeInt(byte[] dest,int destStart,int val)
    {
	dest[destStart+0]=(byte)(val>>>24);
	dest[destStart+1]=(byte)(val>>>16);
	dest[destStart+2]=(byte)(val>>> 8);
	dest[destStart+3]=(byte)(val>>> 0);
    }
    static final public void writeBytes(byte[] dest,int destStart,int len,byte[] src)
    {
	int i,copyLen=len;
	if (copyLen>src.length) copyLen=src.length;
	for (i=0;i<copyLen;i++) dest[destStart+i]=src[i];
	for (i=copyLen;i<len;i++) dest[destStart+i]=0;
    }
    static final public void writeString(byte[] dest,int destStart,int len,String src)
    {
        int i,copyLen=len;
        if (copyLen>src.length()) copyLen=src.length();
        for (i=0;i<copyLen;i++) dest[destStart+i]=(byte)src.charAt(i);
        for (i=copyLen;i<len;i++) dest[destStart+i]=0;   
    }

 //SHA-1 hash utilities:
    private MessageDigest SHA;
    private final int SHA_len=20; //SHA-1 hash is 20 bytes long.
    final private byte[] SHA_makeMessage() { return new byte[64-8-4];}
    private byte[] SHA_digestMessage(byte[] message) {
	SHA.reset();
	SHA.update(message);
	return SHA.digest();
    }
    private byte[] SHA_makeHash(byte[] secretKey,int salt,
			    byte[] header)
    {
	byte[] message=SHA_makeMessage();
	writeBytes(message,0,16,secretKey);
	writeInt(message,16,salt);
	if (header!=null) writeBytes(message,20,16,header);
	byte[] digest=SHA_digestMessage(message);
	//debug("    Created "+digest.length+"-byte SHA-1 hash");
        return digest;
    }
    private boolean SHA_checkHash(byte[] secretKey,int salt,
			    byte[] header,byte[] hash)
    {
	byte[] h2=SHA_makeHash(secretKey,salt,header);
	for (int i=0;i<SHA_len;i++)
	    if (hash[i]!=h2[i])
		return false;
	return true;
    }

 //Authenticate ourselves with this CCS server
    private boolean isAuth=false;
    private byte[] key;
    private int level;
    private java.util.Random rand;
    protected int clientID,clientSalt;
    protected void authenticate(byte[] secretKey) throws IOException
    {
	isAuth=true;
	/*Random number seed depends on key and time*/
	long seed=System.currentTimeMillis();
	key=new byte[16]; /*Key is secret key, zero-padded to 16*/
	for (int i=0;i<secretKey.length;i++) {
	    key[i]=secretKey[i];
	    seed ^= key[i]<<i;
	}
	rand=new java.util.Random(seed);
	for (int i=0;i<10+secretKey.length+(seed&0xf);i++)
	    rand.nextInt();/*Throw away first entries in sequence*/
	try {
	    SHA=MessageDigest.getInstance("SHA");
	} catch(Exception e) {
	    abort("Couldn't load SHA hash code!");
	}
	
	level=0;/*Always ask for security level 0*/
	debug("  Connecting for authentication");
	Socket s=new Socket(hostIP,hostPort);
	debug("  Connected.  Sending request");
	DataOutputStream o=new DataOutputStream(s.getOutputStream());
	DataInputStream i=new DataInputStream(s.getInputStream());
	int s1=rand.nextInt();//My challenge
	int s2=0; //Server's challenge
	try {
	    o.writeInt(0x80000100|level);
	    o.writeInt(s1);
	    s2=i.readInt();
	} catch(IOException e) {
	    abort("Server does not use authentication!");
	}
	byte[] s1hash=new byte[SHA_len];
	try {
	    o.write(SHA_makeHash(key,s2,null));
	    i.readFully(s1hash);
	} catch(EOFException e) {
	    abort("Server does not accept our key!");
	}
	if (!SHA_checkHash(key,s1,null,s1hash)) 
	    abort("Server's key does not match ours (during initial check)!");
	clientID=i.readInt();
	clientSalt=i.readInt();
	debug("  I am client "+clientID+".  My first salt is "+clientSalt+".");
	s.close();
	debug("  Authentication complete");
    }

  
    //Connect to the given CCS server and retrieve parallel program info.
    protected void connect(InetAddress address, int port,byte[] secretKey)
   	throws IOException
    {
   	hostIP=address;
   	hostPort=port;
	
	debug("Connecting...");
	if (secretKey!=null)
	    authenticate(secretKey);

   	sendRequest("ccs_getinfo",0);
	debug("Connected.  Getting machine info.");
   	DataInputStream mach_info=new DataInputStream(
   		  new ByteArrayInputStream(recvResponse())
   		);
	debug("Parsing machine info");
   	//ccs_getinfo returns a list of 4-byte network integers, 
   	// representing the number of nodes and pes for each.
   	numNodes=mach_info.readInt();
   	nodeFirst=new int[numNodes];
   	nodeSize=new int[numNodes];
   	numPes=0;
   	for (int i=0;i<numNodes;i++) {
   		nodeFirst[i]=numPes;
   		numPes+=nodeSize[i]=mach_info.readInt();
	}
    }

    protected static void debug(String s) {
	if (printDebug) System.out.println("CcsServer: "+s);
    }
    protected static void abort(String s) {
	System.out.println("CcsServer FATAL ERROR: "+s);
	System.exit(1);
    }

    protected InetAddress hostIP;//IP address of Converse parallel program
    protected int hostPort;//TCP port number of server socket
    protected int numNodes;//Number of nodes parallel program runs on
    protected int numPes;//Number of PEs parallel program runs on
    protected int nodeFirst[];//Maps node -> # of first processor
    protected int nodeSize[];//Maps node -> # of local processors
    protected static boolean printDebug=false;//Print debugging info

/******** Creation utilities **********/
    /*Parse this base-16 hex string into bytes*/
    public static byte[] parseKey(String s)
    {
	byte[] k=new byte[(s.length()+1)/2];
	for (int i=0;i<s.length();i++) {
	    int shift=16;
	    if (i%2==1) shift=1;
	    k[i/2]+=(byte)(shift*Character.digit(s.charAt(i),16));
	}
	return k;
    }

    /*Convert these command-line arguments into a CCS connection*/
    public static CcsServer create(String args[],boolean withDebug) {
	CcsServer c=null;
	if (args.length<2) {
	    System.out.println("Usage: CcsServer <server DNS name or IP> <port> [ <authentication key>]");
	}
	else {
	    printDebug=withDebug;
	    String host=args[0];
	    int port=0;
	    try {port=Integer.parseInt(args[1]);} 
	    catch (Exception E) {abort("Couldn't parse port number");}
	    byte[] key=null;
	    if (args.length>2) {
		key=parseKey(args[2]);
	    }
	    try {
		c=new CcsServer(host,port,key);
	    } catch(Exception E) {abort("Error connecting to host:"+E);}
	}
	return c;
    }

    /*Trivial example main program-- connects and prints out machine info.*/
    public static void main(String args[]) {
	CcsServer c=CcsServer.create(args,true);
	System.out.println("The CCS server has "+c.getNumPes()+" processors."); 
    }

} // end of CcsServer.java

