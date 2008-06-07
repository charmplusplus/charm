package charj;

import charj.translator.Translator;
import java.io.FileInputStream;
import java.util.Iterator;
import com.martiansoftware.jsap.*;

public class Main 
{
    public static String charmc;
    public static JSAPResult config;
    public static void main(String[] args) throws Exception
    {
        String[] files = processArgs(args);
        for (String filename : files) { 
            System.out.println(Translator.translate(filename, charmc));
        }
    }

    public static String[] processArgs(String[] args) throws Exception
    {
        JSAP processor = new JSAP();
        
        FlaggedOption charmcLocation = new FlaggedOption("charmc")
            .setStringParser(JSAP.STRING_PARSER)
            .setRequired(true)
            .setShortFlag(JSAP.NO_SHORTFLAG)
            .setLongFlag("charmc");
        charmcLocation.setHelp("charm compiler used on generated charm code");
        processor.registerParameter(charmcLocation);

        UnflaggedOption fileList = new UnflaggedOption("files")
            .setStringParser(JSAP.STRING_PARSER)
            .setRequired(true)
            .setGreedy(true);
        fileList.setHelp("A list of Charj (.cj) files to compile");
        processor.registerParameter(fileList);

        config = processor.parse(args);
        if (!config.success()) {
            for (Iterator errs = config.getErrorMessageIterator(); 
                    errs.hasNext();) {
                System.err.println("Error: " + errs.next());
            }
            System.err.println(processor.getHelp());
            System.exit(1);
        }

        charmc = config.getString("charmc");
        String[] files = config.getStringArray("files");
        return files;
    }
}
