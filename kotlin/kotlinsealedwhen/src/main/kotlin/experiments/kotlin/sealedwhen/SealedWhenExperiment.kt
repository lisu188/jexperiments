package experiments.kotlin.sealedwhen

sealed interface Event {
    val id: Int
}

data class Started(override val id: Int, val name: String) : Event

data class Stopped(override val id: Int, val code: Int) : Event

object Heartbeat : Event {
    override val id: Int = -1
}

enum class Severity {
    LOW,
    MEDIUM,
    HIGH
}

fun describe(event: Event): String =
    when (event) {
        is Started -> "started:${event.id}:${event.name}"
        is Stopped -> "stopped:${event.id}:${event.code}"
        Heartbeat -> "heartbeat"
    }

fun severityCode(severity: Severity): Int =
    when (severity) {
        Severity.LOW -> 1
        Severity.MEDIUM -> 2
        Severity.HIGH -> 3
    }

fun main() {
    val events = listOf(Started(1, "compile"), Stopped(2, 0), Heartbeat)
    events.forEach { println(describe(it)) }
    println("severity=${severityCode(Severity.HIGH)}")
}
