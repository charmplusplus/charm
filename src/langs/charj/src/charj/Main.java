package charj;

import charj.translator.Translator;
import java.io.FileInputStream;
import java.util.Iterator;
import com.martiansoftware.jsap.*;

public class Main 
{
    public static String charmc;
    public static boolean debug;
    public static boolean verbose;
    public static boolean stdout;

    public static void main(String[] args) throws Exception
    {
        String[] files = processArgs(args);
        Translator t = new Translator(charmc, debug, verbose);
        for (String filename : files) { 
            if (!stdout) {
                t.translate(filename);
            } else {
                String header = "\n\n" + filename + "\n";
                System.out.println(header + t.translate(filename));
            }
        }
    }

    public static String[] processArgs(String[] args) throws Exception
    {
        JSAP processor = new JSAP();
        
        FlaggedOption _charmc = new FlaggedOption("charmc")
            .setStringParser(JSAP.STRING_PARSER)
            .setRequired(true)
            .setShortFlag(JSAP.NO_SHORTFLAG)
            .setLongFlag("charmc");
        _charmc.setHelp("charm compiler used on generated charm code");
        processor.registerParameter(_charmc);

        Switch _debug = new Switch("debug")
            .setShortFlag(JSAP.NO_SHORTFLAG)
            .setLongFlag("debug");
        _debug.setHelp("enable debugging mode");
        processor.registerParameter(_debug);

        Switch _verbose = new Switch("verbose")
            .setShortFlag('v')
            .setLongFlag("verbose");
        _verbose.setHelp("output extra information");
        processor.registerParameter(_verbose);

        Switch _stdout = new Switch("stdout")
            .setShortFlag(JSAP.NO_SHORTFLAG)
            .setLongFlag("stdout");
        _stdout.setHelp("echo generated code to stdout");
        processor.registerParameter(_stdout);

        UnflaggedOption fileList = new UnflaggedOption("files")
            .setStringParser(JSAP.STRING_PARSER)
            .setRequired(true)
            .setGreedy(true);
        fileList.setHelp("A list of Charj (.cj) files to compile");
        processor.registerParameter(fileList);

        JSAPResult config = processor.parse(args);
        String charmcFlags = "";
        if (!config.success()) {
            for (Iterator errs = config.getErrorMessageIterator(); 
                    errs.hasNext();) {
                System.err.println("Error: " + errs.next());
            }
            System.err.println(processor.getHelp());
            System.exit(1);
        }

        charmc = config.getString("charmc") + charmcFlags;
        debug = config.getBoolean("debug", false);
        verbose = config.getBoolean("verbose", false);
        stdout = config.getBoolean("stdout", false);
        String[] files = config.getStringArray("files");
        return files;
    }
}
