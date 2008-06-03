/***
***/

package charj.translator;

import org.antlr.runtime.*;
import org.antlr.runtime.tree.*;
import org.antlr.stringtemplate.*;
import java.io.*;

public class Translator {

    public static final String templateFile = "src/charj/translator/Charj.stg";
    public static boolean debug = true;
    public static boolean errorCondition = false;

    public static String translate(InputStream io) throws Exception {
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

        // Parse input, creating an AST
        ANTLRInputStream input = new ANTLRInputStream(io);
        CharjLexer lexer = new CharjLexer(input);
        
        // Use lexer tokens to feed tree parser. Note that the parser is a
        // rewriter, so a TokenRewriteStream is needed
        TokenRewriteStream tokens = new TokenRewriteStream(lexer);
        CharjParser parser = new CharjParser(tokens);
        CharjParser.charjSource_return r = parser.charjSource();

        // Walk tree, modifying input buffer
        CommonTree t = (CommonTree)r.getTree();
        CommonTreeNodeStream nodes = new CommonTreeNodeStream(t);
        nodes.setTokenStream(tokens);
        CharjEmitter emitter = new CharjEmitter(nodes);
        emitter.setTemplateLib(templates);
        emitter.charjSource();

        return tokens.toString();
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
