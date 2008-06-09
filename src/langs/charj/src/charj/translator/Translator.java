/***
***/

package charj.translator;

import java.io.*;
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

    // template file locations
    public static final String templateFile = "charj/translator/Charj.stg";

    // variables controlled by command-line arguments
    public String m_charmc;
    public boolean m_debug;
    public boolean m_verbose;
    public boolean m_errorCondition;

    public Translator(
            String _charmc,
            boolean _debug,
            boolean _verbose)
    {
        m_charmc = _charmc;
        m_debug = _debug;
        m_verbose = _verbose;
        m_errorCondition = false;
    }

    public String translate(String filename) throws Exception 
    {
        ANTLRFileStream input = new ANTLRFileStream(filename);
            
        CharjLexer lexer = new CharjLexer(input);
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

    private String translationPass(
            CharjLexer lexer, 
            OutputMode m) throws
        RecognitionException, IOException, InterruptedException
    {
        // Use lexer tokens to feed tree parser
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        CharjParser parser = new CharjParser(tokens);
        CharjParser.charjSource_return r = parser.charjSource();

        // Create node stream for emitters
        CommonTree t = (CommonTree)r.getTree();
        CommonTreeNodeStream nodes = new CommonTreeNodeStream(t);
        nodes.setTokenStream(tokens);

        String output = emit(nodes, m);
        return output;
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

    private String emit(
            CommonTreeNodeStream nodes, 
            OutputMode m) throws
        RecognitionException, IOException, InterruptedException
    {
        CharjEmitter emitter = new CharjEmitter(nodes);
        StringTemplateGroup templates = getTemplates(templateFile);
        emitter.setTemplateLib(templates);
        StringTemplate st = (StringTemplate)emitter.charjSource(m).getTemplate();
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
    
    private void error(
            String sourceName, 
            String msg, 
            CommonTree node) 
    {
        m_errorCondition = true;
        String linecol = ":";
        if ( node!=null ) {
            CommonToken t = (CommonToken)node.getToken();
            linecol = ": line " + t.getLine() + ":" + t.getCharPositionInLine();
        }
        System.err.println(sourceName + linecol + " " + msg);
        System.err.flush();
    }


    private void error(
            String sourceName, 
            String msg) {
        error(sourceName, msg, (CommonTree)null);
    }


    public void error(String msg) {
        error(" [charjc] error", msg, (CommonTree)null);
    }


    public void error(
            String msg, 
            Exception e) {
        error(msg);
        e.printStackTrace(System.err);
    }
}
