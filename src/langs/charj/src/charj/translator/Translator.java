
package charj.translator;

import java.io.*;
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

    // library locations to search for classes
    private String m_stdlib;
    private List<String> m_usrlibs;

    private SymbolTable m_symtab;

    public Translator(
            String _charmc,
            boolean _debug,
            boolean _verbose,
            String _stdlib,
            List<String> _usrlibs)
    {
        m_charmc    = _charmc;
        m_debug     = _debug;
        m_verbose   = _verbose;
        m_stdlib    = _stdlib;
        m_usrlibs   = _usrlibs;
        m_symtab    = new SymbolTable(this);
        m_errorCondition = false;
    }

    public boolean debug()      { return m_debug; }
    public boolean verbose()    { return m_verbose; }

    public static TreeAdaptor m_adaptor = new CommonTreeAdaptor() {
        public Object create(Token token) {
            return new CharjAST(token);
        }
        
        public Object dupNode(Object t) {
            if (t == null) {
                return null;
            }
            return create(((CharjAST)t).token);
        }
    };

    public String translate(String filename) throws Exception 
    {
        ANTLRFileStream input = new ANTLRFileStream(filename);
            
        CharjLexer lexer = new CharjLexer(input);

        semanticPass(lexer);

        input.seek(0);
        String ciOutput = translationPass(lexer, OutputMode.ci);
        writeTempFile(filename, ciOutput, OutputMode.ci);

        input.seek(0);
        String hOutput = translationPass(lexer, OutputMode.h);
        writeTempFile(filename, hOutput, OutputMode.h);
        
        input.seek(0);
        String ccOutput = translationPass(lexer, OutputMode.cc);
        writeTempFile(filename, ccOutput, OutputMode.cc);
        compileTempFiles(filename, m_charmc);

        // Build a string representing all emitted code. This will be printed
        // by the main driver if requested via command-line argument. 
        String ciHeader = "-----CI----------------------------\n";
        String hHeader  = "-----H-----------------------------\n";
        String ccHeader = "-----CC----------------------------\n";
        String footer   = "-----------------------------------\n";
        return ciHeader + ciOutput + hHeader + hOutput + 
            ccHeader + ccOutput + footer;
    }

    private CommonTreeNodeStream prepareNodes(
            CommonTokenStream tokens,
            CharjLexer lexer) throws
        RecognitionException, IOException, InterruptedException
    {
        // Use lexer tokens to feed tree parser
        CharjParser parser = new CharjParser(tokens);
        parser.setTreeAdaptor(m_adaptor);
        CharjParser.charjSource_return r = parser.charjSource();

        // Create node stream for emitters
        CommonTree t = (CommonTree)r.getTree();
        CommonTreeNodeStream nodes = new CommonTreeNodeStream(t);
        nodes.setTokenStream(tokens);
        nodes.setTreeAdaptor(m_adaptor);
        return nodes;
    }

    private String translationPass(
            CharjLexer lexer, 
            OutputMode m) throws
        RecognitionException, IOException, InterruptedException
    {
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        CommonTreeNodeStream nodes = prepareNodes(tokens, lexer);
        nodes.setTokenStream(tokens);
        nodes.setTreeAdaptor(m_adaptor);

        String output = emit(nodes, m);
        return output;
    }

    private ClassSymbol semanticPass(CharjLexer lexer) throws
        RecognitionException, IOException, InterruptedException
    {
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        CommonTreeNodeStream nodes = prepareNodes(tokens, lexer);
        nodes.setTokenStream(tokens);
        nodes.setTreeAdaptor(m_adaptor);

        CharjSemantics sem = new CharjSemantics(nodes);
        ClassSymbol cs = sem.charjSource(m_symtab).cs;
        return cs;
    }

    private String emit(
            CommonTreeNodeStream nodes, 
            OutputMode m) throws
        RecognitionException, IOException, InterruptedException
    {
        CharjEmitter emitter = new CharjEmitter(nodes);
        StringTemplateGroup templates = getTemplates(templateFile);
        emitter.setTemplateLib(templates);
        StringTemplate st = 
            (StringTemplate)emitter.charjSource(m).getTemplate();
        return st.toString();
    }



    private StringTemplateGroup getTemplates(String templateFile)
    {
        StringTemplateGroup templates = null;
        try {
            ClassLoader loader = Thread.currentThread().getContextClassLoader();
            InputStream istream = loader.getResourceAsStream(templateFile);
            BufferedReader reader = 
                new BufferedReader(new InputStreamReader(istream));
            templates = new StringTemplateGroup(reader);
            reader.close();
        } catch(IOException ex) {
            error("Failed to load template file", ex); 
        }
        return templates;
    }

    public File findPackage(String packageName)
    {
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

    /** Load a class from disk looking in lib/package
     *  Side-effect: add class to symtab. This is used by ClassSymbol to
     *  load unknown types from disk. packageName comes from the output
     *  of PackageScope.getFullyQualifiedName
     */
    public ClassSymbol loadType(String packageName, String typeName)
    {
        if (debug()) System.out.println(
                " [charj] loadType(" + typeName + ") from " + packageName);
        
        ClassSymbol cs = null;
        try {
            String packageDir = ".";
            if ( packageName!=null ) {
                packageDir = packageName.replace(".", "/");
            }
            String fullName = packageDir + "/" + typeName + ".cj";
		
            ClassLoader cl = Thread.currentThread().getContextClassLoader();
            boolean fileExists = (cl.getResource(fullName) == null);
            if (!fileExists) {
                if (debug()) System.out.println(
                        " \tloadType(" + typeName + "): not found");
                return null;
            }

            if (debug()) System.out.println(
                    " \tloadType(" + typeName + "): parsing " + 
                    packageName + "." + typeName);
            
            ANTLRInputStream fs = new ANTLRInputStream(
                    cl.getResourceAsStream(fullName));
            fs.name = packageDir + "/" + typeName + ".cj";
            CharjLexer lexer = new CharjLexer(fs);
            
            cs = semanticPass(lexer);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return cs;
    }

    /**
     * Utility function to write a generated .ci or .cc
     * file to disk. Takes a .cj filename and writes a .cc
     * or .ci file to the .charj directory depending on
     * the OutputMode.
     */
    private void writeTempFile(
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
        if (m_verbose) System.out.println(" [charjc] create: " + tempFile);
        FileWriter fw = new FileWriter(tempFile);
        fw.write(output);
        fw.close();
        return;
    }

    /**
     * Enters the .charj directory and compiles the .cc and .ci files 
     * generated from the given filename. The given charmc string 
     * includes all options to be passed to charmc. Any generated .o 
     * file is moved back to the initial directory.
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
        String tempDirectory = baseDirectory + ".charj/";
        String moduleName = filename.substring(lastSlash + 1, lastDot);
        String baseTempFilename = tempDirectory + moduleName;

        // Compile interface file
        String cmd = charmc + " " + baseTempFilename + ".ci";
        File currentDir = new File(".");
        int retVal = exec(cmd, currentDir);
        if (retVal != 0) {
            error("Could not compile generated interface file");
            return;
        }

        // Move decl.h and def.h into temp directory.
        // charmxi/charmc doesn't offer control over where to generate these
        cmd = "mv " + moduleName + ".decl.h " + moduleName + ".def.h " +
            tempDirectory;
        retVal = exec(cmd, currentDir);
         if (retVal != 0) {
            error("Could not move .decl.h and .def.h files " +
                    "into temp directory");
            return;
        }       

        // Compile c++ output
        cmd = charmc + " -c " + baseTempFilename + ".cc" + 
            " -o " + baseTempFilename + ".o";
        retVal = exec(cmd, currentDir);
        if (retVal != 0) {
            error("Could not compile generated C++ file");
            return;
        }

        // move generated .o and .h file into .cj directory
        cmd = "mv -f " + baseTempFilename + ".o " + baseDirectory;
        exec(cmd, currentDir);
        cmd = "cp -f " + baseTempFilename + ".h " + baseDirectory;
        exec(cmd, currentDir);
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
