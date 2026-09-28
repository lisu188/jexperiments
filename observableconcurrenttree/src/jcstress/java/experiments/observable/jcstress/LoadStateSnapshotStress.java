package experiments.observable.jcstress;

import org.openjdk.jcstress.annotations.Actor;
import org.openjdk.jcstress.annotations.Expect;
import org.openjdk.jcstress.annotations.JCStressTest;
import org.openjdk.jcstress.annotations.Outcome;
import org.openjdk.jcstress.annotations.State;
import org.openjdk.jcstress.infra.results.I_Result;

@JCStressTest
@Outcome(id = "2", expect = Expect.ACCEPTABLE, desc = "Reader observed original state.")
@Outcome(id = "3", expect = Expect.ACCEPTABLE, desc = "Reader observed replacement state.")
@Outcome(expect = Expect.FORBIDDEN, desc = "Snapshot must never observe a partially replaced tree.")
@State
public class LoadStateSnapshotStress {
    private final Object tree = TreeReflection.newEmptyTree();

    public LoadStateSnapshotStress() {
        TreeReflection.loadState(tree, TreeReflection.starState(1, 100));
    }

    @Actor
    public void writer() {
        TreeReflection.loadState(tree, TreeReflection.starState(2, 200));
    }

    @Actor
    public void reader(I_Result result) {
        result.r1 = TreeReflection.snapshotSize(tree);
    }
}
