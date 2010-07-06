
package charj.translator;

import org.antlr.stringtemplate.StringTemplate;
import org.antlr.stringtemplate.language.AngleBracketTemplateLexer;

/** Represents a variable definition (name,type) in symbol table (a scope 
 *  thereof)
 */
public class VariableSymbol extends Symbol {
    public boolean isStatic = false;
    public boolean isConst = false;
    public boolean isReadOnly = false;

    public VariableSymbol(
            SymbolTable symtab,
            String name,
            Type type) {
        super(symtab, name, type);
    }

    public String toString() {
        StringBuffer buf = new StringBuffer();
        if ( scope!=null ) {
            buf.append(scope.getScopeName());
            buf.append(".");
        }
        buf.append(name);
        return buf.toString();
    }

    public boolean isPointerType() {
        return this.type instanceof PointerType;
    }

    public String generateInit() {
        if (this.type instanceof PointerType) {
            PointerType pt = (PointerType)this.type;
            StringTemplate st = new StringTemplate
                ("<varname> = new <typename>();", 
                 AngleBracketTemplateLexer.class);
            st.setAttribute("varname", name);
            st.setAttribute("typename", pt.name);
            return st.toString();
        } else {
            return "";
        }
    }

    public String generatePUP() {
        if (isPointerType()) {
            StringTemplate st = new StringTemplate
                ("<name>->pup(p);", 
                 AngleBracketTemplateLexer.class);
            st.setAttribute("name", name);
            return st.toString();
        } else {
            StringTemplate st = new StringTemplate
                ("p | <name>;", 
                 AngleBracketTemplateLexer.class);
            st.setAttribute("name", name);
            return st.toString();
        }
    }
}
