package experiments.kotlin.suspendstate

import kotlin.coroutines.Continuation
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

private var savedContinuation: Continuation<Int>? = null

suspend fun delayedDouble(seed: Int): String {
    val resumed = suspendCoroutine<Int> { continuation ->
        println("kotlin-suspend-point seed=$seed")
        savedContinuation = continuation
    }
    return "delayedDouble=${seed + resumed}"
}

fun resumeSaved(value: Int) {
    val continuation = checkNotNull(savedContinuation) { "no continuation captured" }
    savedContinuation = null
    continuation.resume(value)
}

fun main() {
    val returned = JavaContinuationProbe.callDelayedDouble(10)
    println("java-call-return=${JavaContinuationProbe.describeReturn(returned)}")
    resumeSaved(5)
}
