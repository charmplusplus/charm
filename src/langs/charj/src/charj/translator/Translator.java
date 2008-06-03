/***
***/

package charj.translator;

import org.antlr.runtime.*;
import org.antlr.runtime.tree.*;
import org.antlr.stringtemplate.*;
import java.io.*;

enum OutputMode {
    cc, ci
}

public class Translator {

    public static final String ccTemplateFile = "src/charj/translator/CharjCC.stg";
    public static final String ciTemplateFile = "src/charj/translator/CharjCI.stg";
    public static boolean debug = true;
    public static boolean errorCondition = false;

    public static String translate(String filename) throws Exception {

        ANTLRFileStream input = new ANTLRFileStream(filename);
            
        CharjLexer lexer = new CharjLexer(input);
        String output = translationPass(lexer, OutputMode.cc);
        input.seek(0);
        output += translationPass(lexer, OutputMode.ci);
        
        return output;
    }

    public static String translationPass(CharjLexer lexer, OutputMode m) throws
        RecognitionException, IOException, InterruptedException
    {
        // Use lexer tokens to feed tree parser. Note that the parser is a
        // rewriter, so a TokenRewriteStream is needed
        TokenRewriteStream tokens = new TokenRewriteStream(lexer);
        CharjParser parser = new CharjParser(tokens);
        CharjParser.charjSource_return r = parser.charjSource();

        // Create node stream for emitters
        CommonTree t = (CommonTree)r.getTree();
        CommonTreeNodeStream nodes = new CommonTreeNodeStream(t);
        nodes.setTokenStream(tokens);

        String output = null;
        if (m == OutputMode.cc) {
            output = "\nCC File\n-----------------------\n" + generateCC(nodes);
        } else if (OutputMode.ci == m) {
            output = "\nCI File\n-----------------------\n" + generateCI(nodes);
        }
        return output;
    }

    public static String generateCC(CommonTreeNodeStream nodes) throws
        RecognitionException, IOException, InterruptedException
    {
        CharjCCEmitter emitter = new CharjCCEmitter(nodes);
        StringTemplateGroup templates = getTemplates(ccTemplateFile);
        emitter.setTemplateLib(templates);
        StringTemplate st = (StringTemplate)emitter.charjSource().getTemplate();
        return st.toString();
    }

    public static String generateCI(CommonTreeNodeStream nodes) throws
        RecognitionException, IOException, InterruptedException
    {
        CharjCIEmitter emitter = new CharjCIEmitter(nodes);
        StringTemplateGroup templates = getTemplates(ciTemplateFile);
        emitter.setTemplateLib(templates);
        StringTemplate st = (StringTemplate)emitter.charjSource().getTemplate();
        return st.toString();
    }

    public static StringTemplateGroup getTemplates(String templateFile)
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
    
    public static void error(String sourceName, String msg, CommonTree node) 
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


    public static void error(String sourceName, String msg) {
        error(sourceName, msg, (CommonTree)null);
    }


    public static void error(String msg) {
        error("charj", msg, (CommonTree)null);
    }


    public static void error(String msg, Exception e) {
        error(msg);
        e.printStackTrace(System.err);
    }
}
