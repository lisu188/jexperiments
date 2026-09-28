package experiments.kotlin.valueclassboxing

@JvmInline
value class CustomerId(val raw: String) {
    fun display(): String = "customer:$raw"

    override fun toString(): String = "CustomerId($raw)"
}

class CustomerRegistry {
    fun accept(id: CustomerId): String = "direct=${id.raw}"

    fun acceptNullable(id: CustomerId?): String = "nullable=${id?.raw ?: "missing"}"

    fun collect(ids: List<CustomerId>): String = ids.joinToString(separator = "|") { it.raw }
}

fun main() {
    val id = CustomerId("C-100")
    val registry = CustomerRegistry()

    println(id.display())
    println(registry.accept(id))
    println(registry.acceptNullable(null))
    println(registry.collect(listOf(id, CustomerId("C-200"))))
    println(JavaValueClassCaller.reflectCustomerId("J-300"))
}
