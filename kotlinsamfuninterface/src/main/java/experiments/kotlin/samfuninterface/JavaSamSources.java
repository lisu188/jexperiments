package experiments.kotlin.samfuninterface;

public final class JavaSamSources {
    private JavaSamSources() {
    }

    public static KotlinOperation javaLambda() {
        return value -> value + 10;
    }

    public static KotlinOperation javaMethodReference() {
        return JavaSamSources::triple;
    }

    private static int triple(int value) {
        return value * 3;
    }
}
