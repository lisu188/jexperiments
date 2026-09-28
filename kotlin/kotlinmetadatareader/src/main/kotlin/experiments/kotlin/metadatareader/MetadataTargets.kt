package experiments.kotlin.metadatareader

data class MetadataRecord(
    val name: String,
    val flags: Int = 7
)

sealed class MetadataShape {
    data class Point(val x: Int, val y: Int) : MetadataShape()

    object Empty : MetadataShape()
}

fun metadataTopLevel(record: MetadataRecord): String = "${record.name}:${record.flags}"
