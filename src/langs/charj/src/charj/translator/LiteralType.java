
package charj.translator;

public class LiteralType extends ClassSymbol implements Type {
    public Type baseType;
    public String literal;

    public LiteralType(SymbolTable symtab, Type _baseType) {
        super(symtab, _baseType.getTypeName());
        baseType = _baseType;
    }

    public String getTypeName() {
        return baseType.getTypeName() + "[" + literal + "]";
    }
}
