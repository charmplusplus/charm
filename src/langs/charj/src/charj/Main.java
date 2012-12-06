package charj;

import charj.translator.Translator;
import java.io.FileInputStream;
import java.util.*;
import com.martiansoftware.jsap.*;

public class Main 
{
    public static String m_charmc;
    public static String m_stdlib;
    public static List<String> m_usrlibs;
    public static boolean m_debug;
    public static boolean m_verbose;
    public static boolean m_printAST;
    public static boolean m_stdout;
    public static boolean m_translate_only;
    public static boolean m_count_tokens;
    public static boolean m_executable;

    public static void main(String[] args) throws Exception
    {
        String[] files = processArgs(args);
        Translator t = new Translator(
                m_charmc, 
                m_debug, 
                m_verbose,
                m_printAST,
                m_translate_only,
                m_stdlib,
                m_usrlibs);
        for (String filename : files) { 
            if (m_count_tokens) {
                System.out.println(t.tokenCount(filename));
            } else if (!m_stdout) {
                t.translate(filename);
            } else {
                String header = "\n\n" + filename + "\n";
                System.out.println(header + t.translate(filename));
            }
        }
        if(m_executable)
            t.createExecutable(files, m_charmc);

    }

    public static String[] processArgs(String[] args) throws Exception
    {
        JSAP processor = new JSAP();
        
        FlaggedOption _charmc = new FlaggedOption("charmc")
            .setStringParser(JSAP.STRING_PARSER)
            .setRequired(true)
            .setShortFlag(JSAP.NO_SHORTFLAG)
            .setLongFlag("charmc");
        _charmc.setHelp("Charm compiler used on generated charm code.");
        processor.registerParameter(_charmc);

        FlaggedOption _stdlib = new FlaggedOption("stdlib")
            .setStringParser(JSAP.STRING_PARSER)
            .setRequired(false)
            .setShortFlag(JSAP.NO_SHORTFLAG)
            .setLongFlag("stdlib");
        _stdlib.setHelp("Directory containing the Charj standard libary.");
        processor.registerParameter(_stdlib);

        FlaggedOption _usrlib = new FlaggedOption("usrlib")
            .setStringParser(JSAP.STRING_PARSER)
            .setRequired(false)
            .setShortFlag(JSAP.NO_SHORTFLAG)
            .setLongFlag("lib");
        _usrlib.setHelp("Directories containing user Charj code, " +
                "colon-delimited.");
        processor.registerParameter(_usrlib);

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

        Switch _printAST = new Switch("printAST")
            .setLongFlag("AST");
        _verbose.setHelp("print abstract syntax tree");
        processor.registerParameter(_printAST);

        Switch _stdout = new Switch("stdout")
            .setShortFlag(JSAP.NO_SHORTFLAG)
            .setLongFlag("stdout");
        _stdout.setHelp("echo generated code to stdout");
        processor.registerParameter(_stdout);

        Switch _translate_only = new Switch("translate-only")
            .setShortFlag(JSAP.NO_SHORTFLAG)
            .setLongFlag("translate-only");
        _translate_only.setHelp("translate to C++, but do not compile");
        processor.registerParameter(_translate_only);

        Switch _count_tokens= new Switch("count-tokens")
            .setShortFlag(JSAP.NO_SHORTFLAG)
            .setLongFlag("count-tokens");
        _count_tokens.setHelp("report number of tokens in the input and exit");
        processor.registerParameter(_count_tokens);

        Switch _help = new Switch("help")
            .setShortFlag('h')
            .setLongFlag("help");
        _help.setHelp("Display this help message");
        processor.registerParameter(_help);
        
        Switch _exec = new Switch("exe")
        .setLongFlag("exe");
        _exec.setHelp("call charmc and creates the executable a.out");
        processor.registerParameter(_exec);

        UnflaggedOption fileList = new UnflaggedOption("file")
            .setStringParser(JSAP.STRING_PARSER)
            .setRequired(true)
            .setGreedy(true);
        fileList.setHelp("A list of Charj (.cj) files to compile");
        processor.registerParameter(fileList);

        JSAPResult config = processor.parse(args);

        if (config.getBoolean("help", false)) {
            System.out.println(processor.getHelp());
            System.exit(0);
        }

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
        m_stdlib = config.getString("stdlib");
        m_debug = config.getBoolean("debug", false);
        m_verbose = config.getBoolean("verbose", false);
        m_printAST = config.getBoolean("printAST", false);
        m_stdout = config.getBoolean("stdout", false);
        m_translate_only = config.getBoolean("translate-only", false);
        m_count_tokens = config.getBoolean("count-tokens", false);
        m_executable = config.getBoolean("exe", false);
        
        String usrlib = config.getString("usrlib");
        if (usrlib != null) {
            m_usrlibs = Arrays.asList(usrlib.split(":"));
        } else {
            m_usrlibs = null;
        }

        String[] files = config.getStringArray("file");
        return files;
    }
}
