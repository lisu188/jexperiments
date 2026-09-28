package experiments.kotlin.defaultargs;

public final class JavaDefaultCallers {
    private JavaDefaultCallers() {
    }

    public static String call() {
        KotlinGreeter greeter = new KotlinGreeter("java");
        return KotlinDefaults.label("caller") + "|" + greeter.greet("reader");
    }
}
