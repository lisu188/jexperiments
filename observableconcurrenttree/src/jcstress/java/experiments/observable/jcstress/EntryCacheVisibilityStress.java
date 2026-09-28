package experiments.observable.jcstress;

import org.openjdk.jcstress.annotations.Actor;
import org.openjdk.jcstress.annotations.Expect;
import org.openjdk.jcstress.annotations.JCStressTest;
import org.openjdk.jcstress.annotations.Outcome;
import org.openjdk.jcstress.annotations.State;
import org.openjdk.jcstress.infra.results.I_Result;

@JCStressTest
@Outcome(id = "0", expect = Expect.ACCEPTABLE, desc = "Reader linearized before update.")
@Outcome(id = "1", expect = Expect.ACCEPTABLE, desc = "Reader linearized after update.")
@Outcome(expect = Expect.FORBIDDEN, desc = "Cached immutable Entry must never expose another value.")
@State
public class EntryCacheVisibilityStress {
    private final Object tree;

    public EntryCacheVisibilityStress() {
        tree = TreeReflection.newTree(0, 0);
        TreeReflection.add(tree, 0, 1, 0);
        TreeReflection.valueOf(tree, 1);
    }

    @Actor
    public void writer() {
        TreeReflection.update(tree, 1, 1);
    }

    @Actor
    public void reader(I_Result result) {
        result.r1 = TreeReflection.valueOf(tree, 1);
    }
}
