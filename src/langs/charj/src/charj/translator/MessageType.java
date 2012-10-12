
package charj.translator;

public class MessageType extends Symbol implements Type {
    public Type baseType;

    public MessageType(SymbolTable symtab, Type _baseType) {
        super(symtab, _baseType.getTypeName() + "*@", null);
        baseType = _baseType;
    }

    public String getTypeName() {
        return baseType.getTypeName() + "*@";
    }

    public String getTranslatedTypeName() {
        return baseType.getTypeName() + "*";
    }
}
