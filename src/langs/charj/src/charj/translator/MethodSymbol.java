
package charj.translator;

import org.antlr.stringtemplate.StringTemplate;
import org.antlr.stringtemplate.language.AngleBracketTemplateLexer;

import java.util.List;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;

public class MethodSymbol
    extends SymbolWithScope
    implements Scope {
    /** The enclosing class */
    Scope enclosingScope;

    /** The formal argument list scope */
    LinkedHashMap<String, VariableSymbol> orderedArgs = new LinkedHashMap();

    /** The list of local variables defined anywhere in the method */
    LocalScope locals;
    Map<String, String> entryArgs = new LinkedHashMap();

    public boolean isEntry = false;
    public boolean isStatic = false;
    public boolean isCtor = false;
    public boolean isTraced = false;
    public boolean hasSDAG = false;
    public boolean hasMSA = false;
    public boolean accel = false;

    public Set<VariableSymbol> vars_read = new HashSet<VariableSymbol>();
    public Set<VariableSymbol> vars_written = new HashSet<VariableSymbol>();
    public Set<MethodSymbol> methods_called = new HashSet<MethodSymbol>();

    public Set<CharjAST> accel_ids = new HashSet<CharjAST>();
    public Set<CharjAST> msa_syncs = new HashSet<CharjAST>();
    public Set<CharjAST> msa_accesses = new HashSet<CharjAST>();

    public CharjAST sdagFPL;

    public MethodSymbol(SymbolTable symtab) {
        super(symtab);
    }

    public MethodSymbol(SymbolTable symtab, String name) {
        super(symtab, name);
    }

    public MethodSymbol(
            SymbolTable symtab,
            String name,
            Scope enclosingScope,
            Type retType)
    {
        super(symtab, name);
        this.enclosingScope = enclosingScope;
        this.type = retType;
    }

    public VariableSymbol defineArg(
            String name,
            ClassSymbol type)
    {
        if ( orderedArgs.get(name)!=null ) {
            return null;
        }
        VariableSymbol vs = new VariableSymbol(symtab,name,type);
        define(name, vs);
        return vs;
    }

    public Scope getEnclosingScope()
    {
        return enclosingScope;
    }

    public LocalScope getLocalScope()
    {
        return locals;
    }

    public void setLocalScope(LocalScope s)
    {
        locals = s;
    }

    public String getScopeName()
    {
        return name;
    }

    public Map createMembers()
    {
        if ( orderedArgs==null ) {
            orderedArgs = new LinkedHashMap();
        }
        return orderedArgs;
    }

    public Map getMembers()
    {
        return orderedArgs;
    }

    public String signature()
    {
        return null;
    }

    public String getTraceID()
    {
        // Make sure we don't have any negative or overflow values
        int id = Math.abs(hashCode()/2);
        return String.valueOf(id);
    }

    public String getTraceInitializer()
    {
        StringTemplate st = new StringTemplate(
                "traceRegisterUserEvent(\"<name>\", <id>);",
                AngleBracketTemplateLexer.class);
        st.setAttribute("name", enclosingScope.getScopeName() + "." + name);
        st.setAttribute("id", getTraceID());
        return st.toString();
    }

    public void addAccelIdent(CharjAST ast)
    {
        accel_ids.add(ast);
    }

    public void addMSASync(CharjAST ast)
    {
        msa_syncs.add(ast);
    }

    public void addMSAAccess(CharjAST ast)
    {
        msa_accesses.add(ast);
    }

    public Set<VariableSymbol> readClosure()
    {
        return closure(true, new HashSet<MethodSymbol>());
    }

    public Set<VariableSymbol> writeClosure()
    {
        return closure(false, new HashSet<MethodSymbol>());
    }

    private Set<VariableSymbol> closure(boolean is_read, Set<MethodSymbol> seen)
    {
        Set<VariableSymbol> base_set = is_read ? vars_read : vars_written;
        Set<VariableSymbol> s = new HashSet<VariableSymbol>();
        s.addAll(base_set);
        for (MethodSymbol m : methods_called) {
            if (seen.contains(m)) continue;
            seen.add(m);
            s.addAll(m.closure(is_read, seen));
        }
        return s;
    }

    public Set<MethodSymbol> reachable_methods()
    {
        return reachable_methods_internal(new HashSet<MethodSymbol>());
    }

    private Set<MethodSymbol> reachable_methods_internal(Set<MethodSymbol> seen)
    {
         for (MethodSymbol m : methods_called) {
            if (seen.contains(m)) continue;
            seen.add(m);
            seen.addAll(m.reachable_methods_internal(seen));
         }
        return seen;
    }

    public String toString()
    {
        StringTemplate st = new StringTemplate(
                "<if(entry)>entry <endif><if(parent)><parent>.<endif><name>(<args; separator=\",\">)" +
                "<if(locals)>{<locals; separator=\",\">}<endif>",
                AngleBracketTemplateLexer.class);
        st.setAttribute("entry", isEntry);
        st.setAttribute("parent", enclosingScope != null ? enclosingScope.getScopeName() : null);
        st.setAttribute("name", name);
        st.setAttribute("args", orderedArgs);
        st.setAttribute("locals", locals != null ? locals.getMembers() : null);
        return st.toString();
    }

    public int hashCode()
    {
        return name.hashCode() + orderedArgs.size() + enclosingScope.hashCode();
    }

    /** Two methods are equals() when they have the same name and
     *  the same number of arguments in the same scope.
     */
    public boolean equals(Object object)
    {
        return name.equals(((MethodSymbol)object).name) &&
            orderedArgs.size()==((MethodSymbol)object).orderedArgs.size() &&
            enclosingScope == ((MethodSymbol)object).enclosingScope;
    }

    public String getMangledName()
    {
        String mangled = name;
        boolean isCtor = name.equals(enclosingScope.getScopeName());
        if ( SymbolTable.METHOD_NAMES_TO_MANGLE.contains(name) ||
                (isCtor && SymbolTable.TYPE_NAMES_TO_MANGLE.contains(name)) ) {
            mangled = "cj" + mangled;
        }
        int numargs = getMembers().size();
        if ( numargs > 0 && !isCtor ) {
            mangled += numargs;
        }
        return mangled;
    }

    public void addEntryArg(String type, String name)
    {
        entryArgs.put(type, name);
    }

    public List<String> getEntryArgInitializers()
    {
        List<String> inits = new ArrayList<String>();
        for (Map.Entry<String, String> entry : entryArgs.entrySet()) {
            String name = entry.getValue();
            String type = entry.getKey();
            inits.add(type + "* " + name + " = &__" + name + ";");
        }
        return inits;
    }

}
