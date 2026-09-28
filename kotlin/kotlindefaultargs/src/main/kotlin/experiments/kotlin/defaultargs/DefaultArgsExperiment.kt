package experiments.kotlin.defaultargs

class KotlinGreeter @JvmOverloads constructor(
    private val prefix: String = "hello",
    private val punctuation: String = "!"
) {
    @JvmOverloads
    fun greet(name: String = "bytecode", times: Int = 1): String =
        (1..times).joinToString(separator = "|") { "$prefix $name$punctuation" }
}

object KotlinDefaults {
    @JvmStatic
    @JvmOverloads
    fun label(name: String = "module", version: Int = 1, suffix: String = "ok"): String =
        "$name-$version-$suffix"
}

fun main() {
    val defaultGreeter = KotlinGreeter()
    val customGreeter = KotlinGreeter(prefix = "hi")

    println("kotlin-default-constructor=${defaultGreeter.greet()}")
    println("kotlin-default-method=${customGreeter.greet(times = 2)}")
    println("kotlin-static-default=${KotlinDefaults.label(suffix = "demo")}")
    println("java-overload-call=${JavaDefaultCallers.call()}")
}
