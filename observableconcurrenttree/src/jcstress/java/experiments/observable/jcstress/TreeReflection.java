package experiments.observable.jcstress;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

final class TreeReflection {
    private static final Class<?> TREE = load("ObservableConcurrentTree");
    private static final Class<?> NODE_STATE = load("ObservableConcurrentTree$NodeState");

    private static final Constructor<?> TREE_ROOT = constructor(TREE, Object.class, Object.class);
    private static final Constructor<?> TREE_EMPTY = constructor(TREE);
    private static final Constructor<?> NODE_STATE_CTOR =
            constructor(NODE_STATE, Object.class, Object.class, Object.class);

    private static final Method ADD = method(TREE, "add", Object.class, Object.class, Object.class);
    private static final Method UPDATE = method(TREE, "update", Object.class, Object.class);
    private static final Method MOVE = method(TREE, "move", Object.class, Object.class);
    private static final Method LOAD_STATE = method(TREE, "loadState", java.util.Collection.class);
    private static final Method SNAPSHOT = method(TREE, "snapshot");
    private static final Method GET = method(TREE, "get", Object.class);

    private TreeReflection() {
    }

    static Object newTree(int rootId, int value) {
        return construct(TREE_ROOT, rootId, value);
    }

    static Object newEmptyTree() {
        return construct(TREE_EMPTY);
    }

    static void add(Object tree, int parent, int id, int value) {
        invoke(ADD, tree, parent, id, value);
    }

    static void update(Object tree, int id, int value) {
        invoke(UPDATE, tree, id, value);
    }

    static void move(Object tree, int id, int parent) {
        invoke(MOVE, tree, id, parent);
    }

    static void loadState(Object tree, List<?> state) {
        invoke(LOAD_STATE, tree, state);
    }

    static List<Object> starState(int children, int base) {
        var state = new ArrayList<Object>(children + 1);
        state.add(construct(NODE_STATE_CTOR, 0, null, 0));
        for (int i = 1; i <= children; i++) {
            state.add(construct(NODE_STATE_CTOR, base + i, 0, i));
        }
        return state;
    }

    static int snapshotSize(Object tree) {
        var snapshot = invoke(SNAPSHOT, tree);
        return entries(snapshot).size();
    }

    static int parentOf(Object tree, int nodeId) {
        var snapshot = invoke(SNAPSHOT, tree);
        for (var entry : entries(snapshot)) {
            if (intComponent(entry, "id") == nodeId) {
                var parent = component(entry, "parentId");
                return parent == null ? -1 : (Integer) parent;
            }
        }
        return -2;
    }

    static int valueOf(Object tree, int nodeId) {
        var entry = invoke(GET, tree, nodeId);
        if (entry == null) {
            return Integer.MIN_VALUE;
        }
        return intComponent(entry, "value");
    }

    @SuppressWarnings("unchecked")
    private static List<Object> entries(Object snapshot) {
        return (List<Object>) component(snapshot, "entries");
    }

    private static int intComponent(Object record, String accessor) {
        return (Integer) component(record, accessor);
    }

    private static Object component(Object record, String accessor) {
        return invoke(method(record.getClass(), accessor), record);
    }

    private static Class<?> load(String name) {
        try {
            return Class.forName(name);
        } catch (ClassNotFoundException exception) {
            throw new ExceptionInInitializerError(exception);
        }
    }

    private static Constructor<?> constructor(Class<?> type, Class<?>... parameters) {
        try {
            return type.getConstructor(parameters);
        } catch (ReflectiveOperationException exception) {
            throw new ExceptionInInitializerError(exception);
        }
    }

    private static Method method(Class<?> type, String name, Class<?>... parameters) {
        try {
            return type.getMethod(name, parameters);
        } catch (ReflectiveOperationException exception) {
            throw new ExceptionInInitializerError(exception);
        }
    }

    private static Object construct(Constructor<?> constructor, Object... args) {
        try {
            return constructor.newInstance(args);
        } catch (ReflectiveOperationException exception) {
            throw new IllegalStateException(exception);
        }
    }

    private static Object invoke(Method method, Object receiver, Object... args) {
        try {
            return method.invoke(receiver, args);
        } catch (ReflectiveOperationException exception) {
            var cause = exception.getCause();
            if (cause instanceof RuntimeException runtime) {
                throw runtime;
            }
            throw new IllegalStateException(exception);
        }
    }
}
