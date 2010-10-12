
package charj.translator;

import java.util.Map;
import java.util.List;

public abstract class SymbolWithScope 
    extends Symbol 
    implements Scope {

    public boolean hasSDAG = false;

    public SymbolWithScope(SymbolTable symtab) {
        super(symtab);
    }

    public SymbolWithScope(
            SymbolTable symtab, 
            String name) {
        super(symtab, name, null);
    }

    /** Find any symbol matching name; won't look on disk for classes though */
    public Symbol resolve(String name) {
        Symbol s = null;
        
        // in this scope?
        if (getMembers() != null) {
            s = (Symbol)getMembers().get(name);
        }

        // if not, check any enclosing scope
        if (s == null && getEnclosingScope() != null) {
            s = getEnclosingScope().resolve(name);
        }

        return s;
    }

    /** Scopes other than package and class don't know how to resolve types
     *  (e.g., MethodSymbol).  Look to enclosing scope.
     */
    public Type resolveType(List<TypeName> type) {
        if ( getEnclosingScope()!=null ) {
            return getEnclosingScope().resolveType(type);
        }
        return null;
    }

    public VariableSymbol resolveVariable(String name) {
        Symbol s = getMembers().get(name);
        if (debug()) System.out.println(
                "SymbolWithScope.resolveVariable(" + name + 
                "): examine " + this.getClass().getName() + "=" + toString());
        
        // found it
        if (s != null && s.getClass() == VariableSymbol.class) {
            if (debug()) System.out.println(
                    "SymbolWithScope.resolveVariable(" + name + ") found in " +
                    this.getClass().getName() + "=" + toString());
            return (VariableSymbol)s;
        }

        // not here, check enclosing scope
        if ( s==null && getEnclosingScope() != null ) {
            return getEnclosingScope().resolveVariable(name);
        }

        // not a variable
        if (debug()) System.out.println(
                "SymbolWithScope.resolveVariable(" + name + 
                "): not a variable in " + this.getClass().getName() + 
                "=" + toString());
        return null;
    }

    public MethodSymbol resolveMethod(
            String name, 
            int numargs) {
        if (debug()) System.out.println(
                "SymbolWithScope.resolveMethod(" + name + "," + numargs +
                "): examine " + this.getClass().getName() + "=" + toString());
        
        Symbol s = null;
        if ( numargs == 0 ) {
            s = resolve(name);
        } else {
            s = resolve(name+numargs);
        }

        if ( s!=null && debug() ) System.out.println(
                "SymbolWithScope.resolveMethod(" + name + "," + numargs +
                "): found in context " + this.getClass().getName() +
                "=" + toString());
        else if ( s==null && debug() ) System.out.println(
                "SymbolWithScope.resolveMethod(" + name + "," + numargs +
                "): not found in context " + this.getClass().getName() + 
                "="+toString());
        
        if ( s==null || (s!=null && s.getClass() != MethodSymbol.class) ) {
            // not a method
            if ( s!=null && debug() ) System.out.println(
                    "SymbolWithScope.resolveMethod(" + name + "," + numargs +
                    "): not a method");
            return null;         
        }
        return (MethodSymbol)s;
    }

    /** By default, pass up responsibility in scope hierarchy until we
     *  find a class.
     */
    public boolean isMethod(String name) {
        if ( getEnclosingScope()!=null ) {
            return getEnclosingScope().isMethod(name);
        }
        return false;
    }

    public Symbol remove(String name) {
        Symbol s = (Symbol)getMembers().get(name);
        if ( s==null && getEnclosingScope() != null) {
            return getEnclosingScope().remove(name);
        }
        if ( s != null) {
            getMembers().remove(name);
        }
        return s;
    }

    public Symbol define(
            String name, 
            Symbol sym) {
        // check for error
        Map members = getMembers();
        if ( members == null ) {
            members = createMembers();
        }
        members.put(name, sym);
        sym.scope = this; // track the scope in each symbol
        
        if (debug()) System.out.println(" \tdefine " + name + " as " + sym);
        return sym;
    }

    /** create the member list; we're about to add stuff */
    protected Map<String, Symbol> createMembers() {
        return getMembers();
    }

    /** Scope defaults to just the symbol name; method f's scope is f by 
     *  default. 
     */
    public String getScopeName() {
        return name;
    }

    public String getFullyQualifiedName() {
        String parent = null;
        if ( getEnclosingScope()!=null ) {
            parent = getEnclosingScope().getFullyQualifiedName();
        }
        if ( parent!=null ) {
            return parent + "::" + name;
        }
        return name;
    }

}
