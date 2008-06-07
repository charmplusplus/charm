
package charj.translator;

import java.io.*;
import java.io.InputStream;

/** Support for running processes to compile / exec stuff */
public class StreamEmitter implements Runnable {
    StringBuilder buf = new StringBuilder();
    BufferedReader in;
    Thread sucker;
    PrintStream out;

    public StreamEmitter(InputStream in, PrintStream out) {
        this.in = new BufferedReader( new InputStreamReader(in) );
        this.out = out;
    }
    public void start() {
        sucker = new Thread(this);
        sucker.start();
    }
    public void run() {
        try {
            String line = in.readLine();
            while (line!=null) {
                out.println(line);
                line = in.readLine();
            }
        }
        catch (IOException ioe) {
            System.err.println("can't read output from process");
        }
    }
    /** wait for the thread to finish */
    public void join() throws InterruptedException {
        sucker.join();
    }
}
