package charj;

import charj.translator.Translator;
import java.io.FileInputStream;
import java.util.Iterator;
import com.martiansoftware.jsap.*;

public class Main 
{
    public static String m_charmc;
    public static boolean m_debug;
    public static boolean m_verbose;
    public static boolean m_stdout;

    public static void main(String[] args) throws Exception
    {
        String[] files = processArgs(args);
        Translator t = new Translator(m_charmc, m_debug, m_verbose);
        for (String filename : files) { 
            if (!m_stdout) {
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

        m_charmc = config.getString("charmc") + charmcFlags;
        m_debug = config.getBoolean("debug", false);
        m_verbose = config.getBoolean("verbose", false);
        m_stdout = config.getBoolean("stdout", false);
        String[] files = config.getStringArray("files");
        return files;
    }
}
