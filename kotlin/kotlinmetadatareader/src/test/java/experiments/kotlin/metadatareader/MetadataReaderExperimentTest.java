package experiments.kotlin.metadatareader;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;

import static org.junit.jupiter.api.Assertions.*;

class MetadataReaderExperimentTest {
    @Test
    void kotlinTargetsAndMetadataReaderBranchesAreCovered() throws Exception {
        var record = new MetadataRecord("alpha", 9);
        assertEquals("alpha", record.getName());
        assertEquals(9, record.getFlags());
        assertEquals("alpha:9", MetadataTargetsKt.metadataTopLevel(record));

        var point = new MetadataShape.Point(3, 4);
        assertEquals(3, point.getX());
        assertEquals(4, point.getY());
        assertNotNull(MetadataShape.Empty.INSTANCE);

        var constructor = MetadataReaderExperiment.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        assertNotNull(constructor.newInstance());

        Method printMetadata = MetadataReaderExperiment.class.getDeclaredMethod("printMetadata", Class.class);
        printMetadata.setAccessible(true);
        printMetadata.invoke(null, String.class);
        printMetadata.invoke(null, MetadataRecord.class);

        Method head = MetadataReaderExperiment.class.getDeclaredMethod("head", String[].class);
        head.setAccessible(true);
        assertEquals("<empty>", head.invoke(null, (Object) new String[0]));
        assertEquals("first", head.invoke(null, (Object) new String[]{"first", "second"}));
    }
}
