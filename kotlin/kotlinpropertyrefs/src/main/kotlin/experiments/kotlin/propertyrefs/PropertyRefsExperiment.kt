package experiments.kotlin.propertyrefs

class PropertyBox(initial: String) {
    @JvmField
    var directJvmField: String = "direct-$initial"

    var mutable: String = initial

    val computed: String
        get() = "computed-${mutable.length}"

    lateinit var late: String

    companion object {
        const val CONST_NAME: String = "CONST"

        @JvmField
        val exposedField: String = "exposed"

        @JvmStatic
        fun staticAccessor(): String = "static-$exposedField"
    }
}

object PropertySingleton {
    @JvmField
    val objectField: String = "object-field"

    @JvmStatic
    fun objectCall(): String = "object-call"
}

fun main() {
    val box = PropertyBox("abc")
    box.late = "ready"

    val mutableRef = PropertyBox::mutable
    mutableRef.set(box, "abcd")
    val lateRef = PropertyBox::late

    println("ref=${mutableRef.get(box)} computed=${box.computed} late=${lateRef.get(box)}")
    println("java-property-call=${JavaPropertyCaller.call(box)}")
}
