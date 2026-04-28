package experiments.kotlin.bridges;

public final class JavaPlatformProvider {
    private final String value;

    public JavaPlatformProvider(String value) {
        this.value = value;
    }

    public String maybeText() {
        return value;
    }
}
