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
   public CcsServer(InetAddress address, int port)
   	throws IOException
   {
   	connect(address,port);
   }
   /** Establish a connection with a running Converse program.
    *
    * @param host gives the name of the machine of the running program.
    * @param port gives the TCP port number of the program's server socket.
    */
   public CcsServer(String host,int port)
   	throws IOException, UnknownHostException
   {
   	InetAddress ip=InetAddress.getByName(host);
   	connect(ip,port);
   }
   /** Establish a connection with a running Converse program.
    *
    * @param init gives the CCS initialization string, printed by converse
    *  during startup.
    */
   public CcsServer(String init)
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
   	connect(ip,port);
   }

//-------------------- Send Requests/Recv Responses -----------------
   /** Send a request to a Converse program.  This executes a CCS "handler"
    * (registered in the parallel program with CcsRegisterHandler)
    * on the given processor.
    *
    * @param handlerName gives the name of the CCS handler to process the request.
    * @param destPe gives the (0-based) processor number to process the request.
    */
   public Socket sendRequest(String handlerName, int destPe)
   	throws IOException {return sendRequest(handlerName,destPe,null);}
   /** Send a request to a Converse program.  This executes a CCS "handler"
    * (registered in the parallel program with CcsRegisterHandler)
    * on the given processor with the given data.
    *
    * @param handlerName gives the name of the CCS handler to process the request.
    * @param destPe gives the (0-based) processor number to process the request.
    * @param data gives the data to pass to the handler, if any.
    */
   public Socket sendRequest(String handlerName, int destPe, byte []data)
   	throws IOException
   {
   	//if (replySocket!=null) close(); //Close socket from previous request
   	
   	//Open a socket and send the request header
	debug("  Connecting for request '"+handlerName+"'");
   	replySocket=new Socket(hostIP,hostPort);
	debug("  Connected.  Sending header");
   	DataOutputStream o=new DataOutputStream(replySocket.getOutputStream());
	int dataLen=0;
	if (data!=null) dataLen=data.length;
   	o.writeInt(dataLen);
   	o.writeInt(destPe);
	
   	
   	//Convert the handler name into a flat buffer
	int handlerMAX=32;
   	byte handler[]=new byte[handlerMAX];
   	int i=0;
   	for (i=0;i<handlerName.length();i++)
   		handler[i]=(byte)handlerName.charAt(i);
   	for (i=handlerName.length();i<handlerMAX;i++)
   		handler[i]=0;//Zero-pad handler buffer
   	o.write(handler,0,handlerMAX);
   	debug("  Header sent.  Sending "+dataLen+" bytes of request data");

   	//Send any associated data
   	if (data!=null)
   		o.write(data);
   	o.flush();
   	debug("  Request sent");
	
   	//socket is left open for reply
   	return replySocket;
   }
   /** Wait for a response from Converse program.  This refers to the last
    * executed request, and will wait indefinitely for the response.
    * The response data will be returned as a byte array.
    */
   public byte[] recvResponse() throws IOException
   {return recvResponse(replySocket);}

   /** Wait for a response from Converse program.  This uses the returned value from any
    * previous sendRequest, and will wait indefinitely for the response.
    * The response data will be returned as a byte array.
    */
   public byte[] recvResponse(Socket s) throws IOException
   {
	debug("  Waiting for response");
   	DataInputStream i=new DataInputStream(s.getInputStream());
   	int replyLen=i.readInt(); 
	debug("  Response will be "+replyLen+" bytes");	
   	byte[] reply=new byte[replyLen];
   	if (replyLen!=i.read(reply,0,replyLen))
   		throw new IOException("CCS Reply socket closed early!");
	debug("  Got entire response: "+new String(reply));
   	s.close();
   	return reply;
   }
   
   /** Determine if a response is pending on the given socket
     */
   public boolean hasResponse(Socket s) throws IOException
   {
   	if (s==null) throw new IOException("Null socket!");
   	if (s.getInputStream().available()>0)
   		return true;
   	else
   		return false;
   }
      
   /** Close given CCS request socket.
    */
   public void close(Socket s)
   {
   	if (s!=null) {
   		try {
   			s.close();
		}
		catch (Exception e) { /*ignore*/ }
   		s=null;
   	}
   }
   /** Close current CCS request socket. */
   public void close() {close(replySocket);replySocket=null;}

//----------------- Program Info -------------
   /** Get the number of nodes (address spaces) in parallel machine.*/
   public int getNumNodes() {return numNodes;}
   /** Get the number of processors in parallel machine.*/
   public int getNumPes() {return numPes;}
   /** Get the (0-based) number of the first processor on the given (0-based) node.*/
   public int getNodeFirst(int node) {return nodeFirst[node];}
   /** Get the total number of processors on the given (0-based) node.*/
   public int getNodeSize(int node) {return nodeSize[node];}

   //Connect to the given CCS server and retrieve parallel program info.
   protected void connect(InetAddress address, int port)
   	throws IOException
   {
   	hostIP=address;
   	hostPort=port;
	debug("Connecting...");
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
  
   protected void debug(String s) {
       if (printDebug) System.out.println("CcsServer: "+s);
   }

   protected InetAddress hostIP;//IP address of Converse parallel program
   protected int hostPort;//TCP port number of server socket
   protected int numNodes;//Number of nodes parallel program runs on
   protected int numPes;//Number of PEs parallel program runs on
   protected int nodeFirst[];//Maps node -> # of first processor
   protected int nodeSize[];//Maps node -> # of local processors
   protected Socket replySocket;//Socket to rec'v reply on
   protected boolean printDebug=true;//Print debugging info
} // end of CcsServer.java

