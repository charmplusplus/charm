package charj.translator;

/**
 * Indicates whether we are working on .ci or .cc output.
 */
public enum OutputMode {
    cc(".cc"), 
    ci(".ci"),
    h(".h");

    private final String extension;

    OutputMode(String ext) {
        this.extension = ext;
    }

    public String extension() {
        return extension;
    }
}
