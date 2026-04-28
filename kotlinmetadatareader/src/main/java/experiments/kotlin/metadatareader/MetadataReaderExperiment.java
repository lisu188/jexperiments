package experiments.kotlin.metadatareader;

import java.util.Arrays;
import kotlin.Metadata;

public final class MetadataReaderExperiment {
    private MetadataReaderExperiment() {
    }

    public static void main(String[] args) {
        printMetadata(MetadataRecord.class);
        printMetadata(MetadataShape.class);
        printMetadata(MetadataShape.Point.class);
        printMetadata(MetadataTargetsKt.class);
    }

    private static void printMetadata(Class<?> type) {
        Metadata metadata = type.getAnnotation(Metadata.class);
        if (metadata == null) {
            System.out.println(type.getSimpleName() + ": no kotlin.Metadata");
            return;
        }
        System.out.println(type.getSimpleName()
                + ": kind=" + metadata.k()
                + " mv=" + Arrays.toString(metadata.mv())
                + " d1=" + metadata.d1().length
                + " d2.head=" + head(metadata.d2())
                + " xi=" + metadata.xi()
                + " pn=" + metadata.pn());
    }

    private static String head(String[] values) {
        return values.length == 0 ? "<empty>" : values[0];
    }
}
