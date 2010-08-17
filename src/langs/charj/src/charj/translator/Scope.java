
package charj.translator;

import java.util.Map;
import java.util.List;

public interface Scope {
    
    /** Does this scope have a name?  Call it getScopeName not getName so
     *  things are not confusing in SymbolWithScope subclasses that have a
     *  symbol name and a scope name.
     */
    public String getScopeName();

    public Scope getEnclosingScope();

    public String getFullyQualifiedName();

    /** Return a Map of all contained members */
    public Map<String, Symbol> getMembers();

    /** Look up name in this scope or in enclosing scope if not here;
     *  don't look on disk for classes.
     */
    public Symbol resolve(String name);

    public VariableSymbol resolveVariable(String name);

    /** Look up a typename in this scope or in enclosing scope if not here.
     *  Load from disk if necessary.
     */
    public Type resolveType(List<TypeName> name);

    /** To look up a method, we need to know number of arguments for overloading
     *  so we need separate method to distinguish from resolve().
     *  Don't return variables or classes.
     */
    public MethodSymbol resolveMethod(
            String name, 
            int numargs);

    /** Sometimes we need to test if a name is a method, but we don't know
     *  the full signature (or can't compute easily).  This identifies the
     *  kind of thing 'name' is.
     */
    public boolean isMethod(String name);

    /** Remove a name from this scope */
    public Symbol remove(String name);

    /** Define a symbol in the current scope */
    public Symbol define(
            String name, 
            Symbol sym);
}


