
package charj.translator;

import java.io.*;
import java.nio.*;
import java.nio.channels.*;
import java.nio.charset.*;
import java.util.*;
import org.antlr.runtime.*;
import org.antlr.runtime.tree.*;
import org.antlr.stringtemplate.*;


/**
 * Driver class for lexing, parsing, and output.
 * Takes in file names, parses them and generates code
 * for .ci and .cc files in a .charj directory. Invokes
 * charmc on these outputs and moves any resulting .o file 
 * to the appropriate directory.
 */
public class Translator {

    public static final String templateFile = "charj/translator/Charj.stg";

    // variables controlled by command-line arguments
    private String m_charmc;
    private boolean m_debug;
    private boolean m_verbose;
    private boolean m_errorCondition;
    private boolean m_printAST;
    private boolean m_translate_only;

    // library locations to search for classes
    private String m_stdlib;
    private List<String> m_usrlibs;

    private String m_basename;
    private SymbolTable m_symtab;
    private CommonTree m_ast;
    private CommonTreeNodeStream m_nodes;
    private CommonTokenStream m_tokens;

    public Translator(
            String _charmc,
            boolean _debug,
            boolean _verbose,
            boolean _printAST,
            boolean _translate_only,
            String _stdlib,
            List<String> _usrlibs)
    {
        m_charmc = _charmc;
        m_debug = _debug;
        m_verbose = _verbose;
        m_printAST = _printAST;
        m_translate_only = _translate_only;
        m_stdlib = _stdlib;
        m_usrlibs = _usrlibs;
        m_symtab = new SymbolTable(this);
        m_errorCondition = false;
    }

    public boolean debug()      { return m_debug; }
    public boolean verbose()    { return m_verbose; }
    public String basename()    { return m_basename; }

    public static TreeAdaptor m_adaptor = new CommonTreeAdaptor() {
        public Object create(Token token) {
            return new CharjAST(token);
        }
        
        public Object dupNode(Object t) {
            if (t == null) {
                return null;
            }
            CharjAST orig = (CharjAST)t;
            CharjAST node = (CharjAST)create(orig.token);
            node.def = orig.def;
            node.symbolType = orig.symbolType;
            node.scope = orig.scope;
            return node;
        }
    };

    public int tokenCount(String filename) throws Exception {
        m_basename = filename.substring(0, filename.lastIndexOf("."));
        m_basename = m_basename.substring(m_basename.lastIndexOf("/") + 1);

        ANTLRFileStream input = new ANTLRFileStream(filename);
            
        CharjLexer lexer = new CharjLexer(input);
        m_tokens = new CommonTokenStream(lexer);
        return m_tokens.getTokens().size();
    }

    public String translate(String filename) throws Exception {
        m_basename = filename.substring(0, filename.lastIndexOf("."));
        m_basename = m_basename.substring(m_basename.lastIndexOf("/") + 1);

        ANTLRFileStream input = new ANTLRFileStream(filename);
            
        CharjLexer lexer = new CharjLexer(input);
        m_tokens = new CommonTokenStream(lexer);

        // Use lexer tokens to feed tree parser
        CharjParser parser = new CharjParser(m_tokens);
        parser.setTreeAdaptor(m_adaptor);
        CharjParser.charjSource_return r = parser.charjSource();

        // Create node stream for AST traversals
        m_ast = (CommonTree)r.getTree();
        m_nodes = new CommonTreeNodeStream(m_ast);
        m_nodes.setTokenStream(m_tokens);
        m_nodes.setTreeAdaptor(m_adaptor);

        // do AST rewriting and semantic checking
        if (m_printAST) printAST("Before Preanalysis Pass", "before_preanalysis.html");
        preAnalysisPass();
        if (m_printAST) printAST("After Preanalysis Pass", "after_preanalysis.html");

        resolveTypes();
        if (m_printAST) printAST("After Type Resolution", "after_types.html");

        initPupCollect();
        if (m_printAST) printAST("After Collector Pass", "after_collector.html");

        postAnalysisPass();
        if (m_printAST) printAST("After Postanalysis Pass", "after_postanalysis.html");

		m_nodes = new CommonTreeNodeStream(m_ast);
        m_nodes.setTokenStream(m_tokens);
        m_nodes.setTreeAdaptor(m_adaptor);

        // emit code for .ci, .h, and .cc based on rewritten AST
        String ciOutput = translationPass(OutputMode.ci);
        writeTempFile(filename, ciOutput, OutputMode.ci);

        String hOutput = translationPass(OutputMode.h);
        writeTempFile(filename, hOutput, OutputMode.h);
        
        String ccOutput = translationPass(OutputMode.cc);
        writeTempFile(filename, ccOutput, OutputMode.cc);

        if (!m_translate_only) compileTempFiles(filename, m_charmc);
	
        // Build a string representing all emitted code. This will be printed
        // by the main driver if requested via command-line argument. 
        String ciHeader = "-----CI----------------------------\n";
        String hHeader  = "-----H-----------------------------\n";
        String ccHeader = "-----CC----------------------------\n";
        String footer   = "-----------------------------------\n";
        return ciHeader + ciOutput + hHeader + hOutput + 
            ccHeader + ccOutput + footer;
    }

    private void preAnalysisPass() throws
        RecognitionException, IOException, InterruptedException
    {
        m_nodes.reset();
        CharjPreAnalysis pass = new CharjPreAnalysis(m_nodes);
        pass.setTreeAdaptor(m_adaptor);
        m_ast = (CommonTree)pass.charjSource().getTree();
        m_nodes = new CommonTreeNodeStream(m_ast);
        m_nodes.setTokenStream(m_tokens);
        m_nodes.setTreeAdaptor(m_adaptor);
    }

    private void postAnalysisPass() throws
        RecognitionException, IOException, InterruptedException
    {
        m_nodes.reset();
        CharjPostAnalysis pass = new CharjPostAnalysis(m_nodes);
        pass.setTreeAdaptor(m_adaptor);
        m_ast = (CommonTree)pass.charjSource(m_symtab).getTree();
        m_nodes = new CommonTreeNodeStream(m_ast);
        m_nodes.setTokenStream(m_tokens);
        m_nodes.setTreeAdaptor(m_adaptor);
    }

    private void resolveTypes() throws
        RecognitionException, IOException, InterruptedException
    {
        m_nodes.reset();
        if (m_verbose) System.out.println("\nDefiner Phase\n----------------");
        SymbolDefiner definer = new SymbolDefiner(m_nodes, m_symtab);
        definer.downup(m_ast);
        m_nodes.reset();
        definer.downup(m_ast);
        if (m_verbose) System.out.println("\nResolver Phase\n----------------");
        if (m_printAST) printAST("After Type Definition", "after_definition.html");
        m_nodes.reset();
        SymbolResolver resolver = new SymbolResolver(m_nodes, m_symtab);
        resolver.downup(m_ast);
    }

    private void initPupCollect() throws
        RecognitionException, IOException, InterruptedException
    {
        m_nodes.reset();
        if (m_verbose) System.out.println("\nInitPupCollector Phase\n----------------");
        InitPUPCollector collector = new InitPUPCollector(m_nodes);
        collector.downup(m_ast);
    }

    private String translationPass(OutputMode m) throws
        RecognitionException, IOException, InterruptedException
    {
        m_nodes.reset();
        CharjEmitter emitter = new CharjEmitter(m_nodes);
        StringTemplateGroup templates = getTemplates(templateFile);
        emitter.setTemplateLib(templates);
        StringTemplate st = 
            (StringTemplate)emitter.charjSource(m_symtab, m).getTemplate();
        return st.toString();
    }

    public static StringTemplateGroup getTemplates(String templateFile) {
        StringTemplateGroup templates = null;
        try {
            ClassLoader loader = Thread.currentThread().getContextClassLoader();
            InputStream istream = loader.getResourceAsStream(templateFile);
            BufferedReader reader = 
                new BufferedReader(new InputStreamReader(istream));
            templates = new StringTemplateGroup(reader);
            reader.close();
        } catch(IOException ex) {
            System.err.println(ex.getMessage());
            ex.printStackTrace(System.err);
        }
        return templates;
    }

    public File findPackage(String packageName) {
        String packageDir = packageName.replace(".", "/");
        File p = new File(packageDir);
        if ( debug() ) System.out.println(
                " [charj] findPackage " + packageName + 
                " trying " + p.getAbsoluteFile());
       
        // check current directory
        if ( p.exists() ) {
            return p;
        }

        // look in user libs if any
        if ( m_usrlibs != null ) {
            for (String lib : m_usrlibs) {
                p = new File(lib, packageDir);
                if (debug() ) System.out.println(
                        " \tnot found, now trying " + p.getAbsoluteFile());
                if ( p.exists() ) {
                    return p;
                }
            }
        }

        // look in standard lib
        p = new File(m_stdlib, packageDir);
        if ( debug() ) System.out.println(
                " \tnot found, now trying " + p.getAbsoluteFile());
        if ( p.exists() ) {
            return p;
        }

        return null;
    }

    /**
     * Read the given file name in as a string.
     */
    public static String readFile(String path) throws IOException {
      FileInputStream stream = new FileInputStream(new File(path));
      try {
        FileChannel fc = stream.getChannel();
        MappedByteBuffer bb = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size());
        return Charset.defaultCharset().decode(bb).toString();
      }
      finally {
        stream.close();
      }
    }

    /**
     * Utility function to write a generated .ci, .cc, or .h
     * file to disk. Takes a .cj filename and writes a .cc,
     * .ci, or .h file to the .charj directory depending on
     * the OutputMode.
     */
    private void writeTempFile(
            String filename, 
            String output,
            OutputMode m) throws
        IOException
    {
        int lastDot = filename.lastIndexOf(".");
        filename = filename.substring(0, lastDot) + m.extension();
        writeTempFile(filename, output);
        return;
    }

    private void writeTempFile(
            String filename,
            String output) throws
        IOException
    {
        if (m_verbose) System.out.println(" [charjc] create: " + filename);
        FileWriter fw = new FileWriter(filename);
        fw.write(output);
        fw.close();
        return;
    }

    /**
     * Compiles the .cc and .ci files generated from the given filename.
     * The given charmc string includes all options to be passed to charmc.
     * Any generated .o file is moved back to the initial directory.
     */
    private void compileTempFiles(
            String filename,
            String charmc) throws
        IOException, InterruptedException
    {
        int lastDot = filename.lastIndexOf(".");
        int lastSlash = filename.lastIndexOf("/");
        String baseDirectory = filename.substring(0, lastSlash + 1);
        if (baseDirectory.equals("")) {
            baseDirectory = "./";
        }
        String moduleName = filename.substring(lastSlash + 1, lastDot);
        String baseTempFilename = moduleName;

        // Compile interface file
        String cmd = charmc + " " + baseTempFilename + ".ci";
        File currentDir = new File(".");
        int retVal = exec(cmd, currentDir);
        if (retVal != 0) {
            error("Could not compile generated interface file.");
            return;
        }

        // Move decl.h and def.h into temp directory.
        // charmxi/charmc doesn't offer control over where to generate these
        cmd = "touch " + baseTempFilename + ".decl.h " +
            baseTempFilename + ".def.h";
        retVal = exec(cmd, currentDir);
        if (retVal != 0) {
            error("Could not touch .decl.h and .def.h files.");
            return;
        }

        // Compile c++ output
        cmd = charmc + " -I" + m_stdlib + "/charj/libs -c " +
            baseTempFilename + ".cc" + " -o " + baseTempFilename + ".o";
        retVal = exec(cmd, currentDir);
        if (retVal != 0) {
            error("Could not compile generated C++ file");
            return;
        }

    }
    /**
    * Create the executable file from the generated .o files
    */
    public void createExecutable(String[] files, String charmc) throws
	IOException, InterruptedException
    {
	// Create the executable
	File currentDir = new File(".");        
        String cmd = charmc + "-language charm++ -o a.out ";
	for(String filename : files){
	    int lastDot = filename.lastIndexOf(".");
            int lastSlash = filename.lastIndexOf("/");
            String baseDirectory = filename.substring(0, lastSlash + 1);
            if (baseDirectory.equals("")) {
                baseDirectory = "./";
            }
            String baseTempFilename = filename.substring(lastSlash + 1, lastDot);
 	    cmd += baseTempFilename + ".o ";
	}
        int retVal = exec(cmd, currentDir);
        if (retVal != 0) {
            error("Could not create the executable");
            return;
        }
    }
	
    /**
     * Utility function to execute a given command line.
     */
    private int exec(String cmd, File outputDir) throws
    	IOException, InterruptedException
    {
        if (m_verbose) System.out.println(" [charjc] exec: " + cmd);
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

    /**
     * Print a representation of the Charj AST. If message is not null,
     * it is printed, along with an ASCII representation of the tree,
     * to stdout. If filename is not null, an html temp file containin
     * the representation is printed to filename.
     */
    public void printAST(String message, String filename) throws IOException
    {
        if (filename != null) {
            ASTHTMLPrinter htmlPrinter = new ASTHTMLPrinter();
            TreeTraverser.visit((CharjAST)m_ast, htmlPrinter);
            writeTempFile(filename, htmlPrinter.output());
        }

        if (message != null) {
            String header = "----------\n" + "AST: " + message + "\n----------\n";
            String footer = "\n----------\n";
            String body = null;
            if (m_ast != null) {
                body = m_ast.toStringTree();
            } else {
                body = "Null tree, no AST available";
            }
            System.out.println(header + body + footer);
        }
    }
    
    public void error(
            BaseRecognizer recog, 
            String msg, 
            CharjAST node) 
    {
        String sourceName = "";
        if (recog == null) {
            sourceName = "<anonymous>";
        } else {
            sourceName = recog.getSourceName();
        }
        error(sourceName, msg, node);
    } 

    private void error(
            String sourceName, 
            String msg, 
            CharjAST node) 
    {
        m_errorCondition = true;
        String linecol = ":";
        if ( node!=null ) {
            CommonToken t = (CommonToken)node.getToken();
            linecol = ": line " + t.getLine() + ":" + 
                t.getCharPositionInLine();
        }
        System.err.println(sourceName + linecol + " " + msg);
        System.err.flush();
    }

    private void error(
            String sourceName, 
            String msg) {
        error(sourceName, msg, (CharjAST)null);
    }


    public void error(String msg) {
        error(" [charjc] error", msg, (CharjAST)null);
    }


    public void error(
            String msg, 
            Exception e) {
        error(msg);
        e.printStackTrace(System.err);
    }
}
