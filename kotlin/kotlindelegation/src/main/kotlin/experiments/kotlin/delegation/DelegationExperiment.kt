package experiments.kotlin.delegation

import kotlin.reflect.KProperty

interface Reporter {
    fun report(message: String): String
}

class PrefixReporter(private val prefix: String) : Reporter {
    override fun report(message: String): String = "$prefix:$message"
}

class ReportingService(delegate: Reporter) : Reporter by delegate {
    fun ownReport(): String = report("own")
}

class AuditPropertyDelegate(private var value: String) {
    operator fun getValue(thisRef: Any?, property: KProperty<*>): String {
        println("getValue(${property.name})")
        return value
    }

    operator fun setValue(thisRef: Any?, property: KProperty<*>, newValue: String) {
        println("setValue(${property.name},$newValue)")
        value = newValue
    }
}

class DelegatedProperties {
    var status: String by AuditPropertyDelegate("new")
    val summary: String by lazy { "summary-$status" }
}

fun main() {
    val service = ReportingService(PrefixReporter("delegate"))
    println(service.report("call"))
    println(service.ownReport())

    val properties = DelegatedProperties()
    println("initial=${properties.status}")
    properties.status = "ready"
    println("summary=${properties.summary}")
}
