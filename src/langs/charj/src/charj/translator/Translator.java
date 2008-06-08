/***
***/

package charj.translator;

import java.io.*;
import org.antlr.runtime.*;
import org.antlr.runtime.tree.*;
import org.antlr.stringtemplate.*;
import charj.translator.StreamEmitter;

enum OutputMode {
    cc(".cc"), 
    ci(".ci");

    private final String extension;

    OutputMode(String ext) {
        this.extension = ext;
    }

    public String extension() {
        return extension;
    }
}

public class Translator {

    public static final String ccTemplateFile = 
        "charj/translator/CharjCC.stg";
    public static final String ciTemplateFile = 
        "charj/translator/CharjCI.stg";
    public static boolean debug = true;
    public static boolean errorCondition = false;

    public static String translate(
            String filename,
            String charmc) throws Exception 
    {
        ANTLRFileStream input = new ANTLRFileStream(filename);
            
        CharjLexer lexer = new CharjLexer(input);
        String output = translationPass(lexer, OutputMode.cc);
        writeTempFile(filename, output, OutputMode.cc);
        input.seek(0);
        output = translationPass(lexer, OutputMode.ci);

        writeTempFile(filename, output, OutputMode.ci);
        compileTempFiles(filename, charmc);
        
        return output;
    }

    private static String translationPass(
            CharjLexer lexer, 
            OutputMode m) throws
        RecognitionException, IOException, InterruptedException
    {
        // Use lexer tokens to feed tree parser. Note that the parser is a
        // rewriter, so a TokenRewriteStream is needed
        //TokenRewriteStream tokens = new TokenRewriteStream(lexer);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        CharjParser parser = new CharjParser(tokens);
        CharjParser.charjSource_return r = parser.charjSource();

        // Create node stream for emitters
        CommonTree t = (CommonTree)r.getTree();
        CommonTreeNodeStream nodes = new CommonTreeNodeStream(t);
        nodes.setTokenStream(tokens);

        String output = null;
        if (m == OutputMode.cc) {
            output = generateCC(nodes);
        } else if (OutputMode.ci == m) {
            output = generateCI(nodes);
        }
        return output;
    }

    private static void writeTempFile(
            String filename, 
            String output,
            OutputMode m) throws
        IOException
    {
        int lastDot = filename.lastIndexOf(".");
        int lastSlash = filename.lastIndexOf("/");
        String tempFile = filename.substring(0, lastSlash + 1) + ".charj/";
        new File(tempFile).mkdir();
        tempFile += filename.substring(lastSlash + 1, lastDot) + m.extension();
        FileWriter fw = new FileWriter(tempFile);
        fw.write(output);
        fw.close();
        return;
    }

    private static void compileTempFiles(
            String filename,
            String charmc) throws
        IOException, InterruptedException
    {
        int lastDot = filename.lastIndexOf(".");
        int lastSlash = filename.lastIndexOf("/");
        String baseFilename = filename.substring(0, lastSlash + 1) + 
            ".charj/" + filename.substring(lastSlash + 1, lastDot);
        String cmd = charmc + " " + baseFilename + ".ci";
        File currentDir = new File(".");
        int retVal = exec(cmd, currentDir);
        if (retVal != 0) return;
        
        cmd = charmc + " -c " + baseFilename + ".cc";
        retVal = exec(cmd, currentDir);
        if (retVal != 0) return;

        cmd = "mv -f " + baseFilename + ".o" + " .";
        exec(cmd, currentDir);
    }

    private static int exec(String cmd, File outputDir) throws
        IOException, InterruptedException
    {
        System.out.println("exec: " + cmd);
        Process p = Runtime.getRuntime().exec(cmd, null, outputDir);
        StreamEmitter stdout = new StreamEmitter(
                p.getInputStream(), System.out);
        StreamEmitter stderr = new StreamEmitter(
                p.getErrorStream(), System.err);
        stdout.start();
        stderr.start();
        p.waitFor();
        stdout.join();
        stderr.join();
        int retVal = p.exitValue();
        return retVal;
    }

    private static String generateCC(CommonTreeNodeStream nodes) throws
        RecognitionException, IOException, InterruptedException
    {
        CharjCCEmitter emitter = new CharjCCEmitter(nodes);
        StringTemplateGroup templates = getTemplates(ccTemplateFile);
        emitter.setTemplateLib(templates);
        StringTemplate st = (StringTemplate)emitter.charjSource().getTemplate();
        return st.toString();
    }

    private static String generateCI(CommonTreeNodeStream nodes) throws
        RecognitionException, IOException, InterruptedException
    {
        CharjCIEmitter emitter = new CharjCIEmitter(nodes);
        StringTemplateGroup templates = getTemplates(ciTemplateFile);
        emitter.setTemplateLib(templates);
        StringTemplate st = (StringTemplate)emitter.charjSource().getTemplate();
        return st.toString();
    }

    private static StringTemplateGroup getTemplates(String templateFile)
    {
        StringTemplateGroup templates = null;
        try {
            ClassLoader loader = Thread.currentThread().getContextClassLoader();
            InputStream istream = loader.getResourceAsStream(templateFile);
            BufferedReader reader = new BufferedReader(new InputStreamReader(istream));
            templates = new StringTemplateGroup(reader);
            reader.close();
        } catch(IOException ex) {
            error("Failed to load template file", ex); 
        }
        return templates;
    }
    
    private static void error(
            String sourceName, 
            String msg, 
            CommonTree node) 
    {
        errorCondition = true;
        String linecol = ":";
        if ( node!=null ) {
            CommonToken t = (CommonToken)node.getToken();
            linecol = "line " + t.getLine() + ":" + t.getCharPositionInLine();
        }
        System.err.println(sourceName + ": " + linecol + " " + msg);
        System.err.flush();
    }


    private static void error(
            String sourceName, 
            String msg) {
        error(sourceName, msg, (CommonTree)null);
    }


    public static void error(String msg) {
        error("charj", msg, (CommonTree)null);
    }


    public static void error(
            String msg, 
            Exception e) {
        error(msg);
        e.printStackTrace(System.err);
    }
}
