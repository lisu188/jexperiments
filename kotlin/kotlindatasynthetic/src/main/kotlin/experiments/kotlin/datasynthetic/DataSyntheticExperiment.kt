package experiments.kotlin.datasynthetic

data class BuildCoordinate(
    val group: String,
    val artifact: String,
    val version: Int = 1
) {
    fun gav(): String = "$group:$artifact:$version"
}

fun main() {
    val coordinate = BuildCoordinate(group = "experiments", artifact = "data")
    val copy = coordinate.copy(version = 2)

    println("component1=${coordinate.component1()}")
    println("copy=$copy")
    println("equals=${coordinate == copy}")
    println("java-data-call=${JavaDataCaller.callDataApi()}")
}
