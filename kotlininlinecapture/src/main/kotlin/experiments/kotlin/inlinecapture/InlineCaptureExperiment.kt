package experiments.kotlin.inlinecapture

inline fun <reified T> renderValue(
    value: Any,
    prefix: String,
    noinline deferred: (String) -> String,
    crossinline immediate: (T) -> String
): String {
    val typed = if (value is T) immediate(value) else "not-${T::class.java.simpleName}"
    val savedForLater = listOf(deferred).single()
    return "${T::class.java.simpleName}:$typed:${savedForLater(prefix)}"
}

inline fun runLater(crossinline block: () -> String): Runnable =
    Runnable { println("crossinline-runnable=${block()}") }

fun main() {
    val captured = "capture"
    val rendered = renderValue<String>(
        value = "kotlin",
        prefix = "prefix",
        deferred = { text -> "$text-$captured" },
        immediate = { text -> text.uppercase() + "-$captured" }
    )

    println("rendered=$rendered")
    runLater { "inside-$captured" }.run()

    val noinlineFunction: (String) -> String = { text -> "allocated-$text-$captured" }
    println(noinlineFunction("lambda"))
}
