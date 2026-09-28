package experiments.observable.jcstress;

import org.openjdk.jcstress.annotations.Actor;
import org.openjdk.jcstress.annotations.Expect;
import org.openjdk.jcstress.annotations.JCStressTest;
import org.openjdk.jcstress.annotations.Outcome;
import org.openjdk.jcstress.annotations.State;
import org.openjdk.jcstress.infra.results.I_Result;

@JCStressTest
@Outcome(id = "1", expect = Expect.ACCEPTABLE, desc = "Snapshot linearized before move.")
@Outcome(id = "2", expect = Expect.ACCEPTABLE, desc = "Snapshot linearized after move.")
@Outcome(expect = Expect.FORBIDDEN, desc = "Node must never have a missing or inconsistent parent.")
@State
public class MoveSnapshotStress {
    private final Object tree;

    public MoveSnapshotStress() {
        tree = TreeReflection.newTree(0, 0);
        TreeReflection.add(tree, 0, 1, 1);
        TreeReflection.add(tree, 0, 2, 2);
        TreeReflection.add(tree, 1, 3, 3);
    }

    @Actor
    public void mover() {
        TreeReflection.move(tree, 3, 2);
    }

    @Actor
    public void reader(I_Result result) {
        result.r1 = TreeReflection.parentOf(tree, 3);
    }
}
