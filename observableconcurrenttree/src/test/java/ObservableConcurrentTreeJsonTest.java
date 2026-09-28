import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;

import org.junit.jupiter.api.Test;

class ObservableConcurrentTreeJsonTest {
    record Payload(String name, long timestamp, List<String> tags) {
    }

    @Test
    void roundTripsUnicodeNullValuesAndVersion() throws Exception {
        var tree = new ObservableConcurrentTree<String, String>("korzeń", "ąęćłńóśżź 日本語 😀");
        tree.add("korzeń", "null-value", null);
        tree.add("korzeń", "child", "wartość");
        var version = tree.getVersion();

        var restored = ObservableConcurrentTree.fromJson(
                tree.toJson(),
                String.class,
                String.class);

        assertEquals(tree.snapshot(), restored.snapshot());
        assertEquals(version, restored.getVersion());
        TreeTestSupport.assertValid(restored);
    }

    @Test
    void supportsRecordValuesWithCallerMapper() throws Exception {
        var mapper = new ObjectMapper();
        var tree = new ObservableConcurrentTree<String, Payload>(
                "root",
                new Payload("root", 1L, List.of("a")));
        tree.add("root", "child", new Payload("child", 2L, List.of("x", "y")));

        var restored = ObservableConcurrentTree.fromJson(
                mapper,
                tree.toJson(mapper),
                String.class,
                Payload.class);

        assertEquals(tree.snapshot(), restored.snapshot());
    }

    @Test
    void rejectsMalformedOrStructurallyInvalidJson() {
        assertThrows(Exception.class,
                () -> ObservableConcurrentTree.fromJson("{", Integer.class, Integer.class));
        assertThrows(Exception.class,
                () -> ObservableConcurrentTree.fromJson(
                        "{\"version\":1,\"nodes\":[{\"id\":1,\"parentId\":99,\"value\":1}]}",
                        Integer.class,
                        Integer.class));
        assertThrows(Exception.class,
                () -> ObservableConcurrentTree.fromJson(
                        "{\"version\":-1,\"nodes\":[]}",
                        Integer.class,
                        Integer.class));
    }

    @Test
    void jsonDoesNotContainImplementationOrObserverMetadata() throws Exception {
        var tree = new ObservableConcurrentTree<Integer, String>(0, "root");
        tree.addObserver(change -> { });
        var json = tree.toJson();

        assertTrue(json.contains("\"version\""));
        assertTrue(json.contains("\"nodes\""));
        org.junit.jupiter.api.Assertions.assertFalse(json.contains("observer"));
        org.junit.jupiter.api.Assertions.assertFalse(json.contains("ReentrantReadWriteLock"));
    }
}
