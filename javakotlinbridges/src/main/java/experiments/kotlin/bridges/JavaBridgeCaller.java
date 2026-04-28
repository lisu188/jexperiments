package experiments.kotlin.bridges;

import java.util.Arrays;

public final class JavaBridgeCaller {
    private JavaBridgeCaller() {
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    public static String call() {
        StringKotlinBase typed = new StringKotlinBase();
        KotlinBase raw = typed;
        Object bridge = raw.identity("bridge");
        OverloadEdges edges = new OverloadEdges();
        return typed.identity("java")
                + "|" + bridge
                + "|" + BridgeExperimentKt.joinStrings(Arrays.asList("x", "y"))
                + "|" + edges.acceptNullableName(null)
                + "|" + edges.accept(7);
    }
}
