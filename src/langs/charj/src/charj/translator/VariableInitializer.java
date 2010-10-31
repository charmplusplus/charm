package charj.translator;

import org.antlr.runtime.*;
import org.antlr.runtime.tree.*;
import org.antlr.stringtemplate.*;
import org.antlr.stringtemplate.StringTemplate;
import org.antlr.stringtemplate.language.AngleBracketTemplateLexer;

/** 
 *  Encapsulates a "variableInitializer" in the CharjAST, used for sending the
 *  reference of the AST to the emitter. 
 */
public class VariableInitializer {
    public CharjAST init;
    private CharjAST ident;

    public VariableInitializer(CharjAST init_, CharjAST ident_) {
        init = init_;
        ident = ident_;
    }
    
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

    public String emit() {
        CommonTree m_ast = (CommonTree)init;
        CommonTreeNodeStream m_nodes = new CommonTreeNodeStream(m_ast);
        m_nodes.setTreeAdaptor(m_adaptor);
        CharjEmitter emitter = new CharjEmitter(m_nodes);
        emitter.setTemplateLib(Translator.
                               getTemplates(Translator.templateFile));
        try {
            StringTemplate st = 
                (StringTemplate)emitter.expression().getTemplate();

            StringTemplate st2 = new StringTemplate
                ("<name> = <init>;", AngleBracketTemplateLexer.class);

            st2.setAttribute("name", ident.def.name);
            st2.setAttribute("init", st.toString());

            return st2.toString();
        } catch (RecognitionException ex) {
            //TODO: Decouple Translator's error handling and mix with this
            System.err.println(ex.getMessage());
            ex.printStackTrace(System.err);
            return "";
        }
    }
}