package experiments.kotlin.bridges

open class KotlinBase<T> {
    open fun identity(value: T): T = value
}

class StringKotlinBase : KotlinBase<String>() {
    override fun identity(value: String): String = value.uppercase()
}

interface Producer<out T> {
    fun produce(): T
}

class StringProducer(private val value: String) : Producer<String> {
    override fun produce(): String = value
}

@JvmName("joinStrings")
fun join(values: List<String>): String = values.joinToString(separator = ":")

@JvmName("joinInts")
fun join(values: List<Int>): String = values.joinToString(separator = ":")

@JvmSynthetic
fun kotlinOnlySecret(): String = "hidden-from-java-source"

class OverloadEdges {
    @JvmName("acceptNullableName")
    fun accept(name: String?): String = "name=${name ?: "null"}"

    fun accept(count: Int): String = "count=$count"
}

fun platformLength(provider: JavaPlatformProvider): Int = provider.maybeText().length

fun main() {
    val typed = StringKotlinBase()
    val producer: Producer<String> = StringProducer("produced")

    println("override=${typed.identity("kotlin")}")
    println("producer=${producer.produce()}")
    println("join=${join(listOf("a", "b"))}|${join(listOf(1, 2))}")
    println("platformLength=${platformLength(JavaPlatformProvider("platform"))}")
    println("java-bridge-call=${JavaBridgeCaller.call()}")
}
