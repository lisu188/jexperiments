package experiments.kotlin.samfuninterface

fun interface KotlinOperation {
    fun apply(value: Int): Int
}

class OperationRunner {
    fun run(label: String, operation: KotlinOperation, value: Int): String =
        "$label=${operation.apply(value)}"
}

fun double(value: Int): Int = value * 2

fun kotlinLambda(): KotlinOperation = KotlinOperation { value -> value + 1 }

fun kotlinMethodReference(): KotlinOperation = KotlinOperation(::double)

fun main() {
    val runner = OperationRunner()

    println(runner.run("kotlin-lambda", kotlinLambda(), 3))
    println(runner.run("kotlin-method-reference", kotlinMethodReference(), 3))
    println(runner.run("java-lambda", JavaSamSources.javaLambda(), 3))
    println(runner.run("java-method-reference", JavaSamSources.javaMethodReference(), 3))
}
