package com.lis.distributed.thread.pool;

import org.openjdk.jcstress.annotations.Actor;
import org.openjdk.jcstress.annotations.Arbiter;
import org.openjdk.jcstress.annotations.Expect;
import org.openjdk.jcstress.annotations.JCStressTest;
import org.openjdk.jcstress.annotations.Outcome;
import org.openjdk.jcstress.annotations.State;
import org.openjdk.jcstress.infra.results.JJ_Result;

@JCStressTest
@Outcome(id = {"0, 1", "1, 0"}, expect = Expect.ACCEPTABLE)
@State
public class RepositoryRequestIdStress {
    private final DataRepository repository = new DataRepository();

    @Actor
    public void first(JJ_Result result) {
        result.r1 = repository.register().id();
    }

    @Actor
    public void second(JJ_Result result) {
        result.r2 = repository.register().id();
    }

    @Arbiter
    public void cleanup() {
        repository.close();
    }
}
