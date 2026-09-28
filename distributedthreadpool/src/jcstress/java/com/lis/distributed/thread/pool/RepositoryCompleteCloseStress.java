package com.lis.distributed.thread.pool;

import org.openjdk.jcstress.annotations.Actor;
import org.openjdk.jcstress.annotations.Arbiter;
import org.openjdk.jcstress.annotations.Expect;
import org.openjdk.jcstress.annotations.JCStressTest;
import org.openjdk.jcstress.annotations.Outcome;
import org.openjdk.jcstress.annotations.State;
import org.openjdk.jcstress.infra.results.II_Result;

@JCStressTest
@Outcome(id = {"1, 0", "-1, 0"}, expect = Expect.ACCEPTABLE)
@State
public class RepositoryCompleteCloseStress {
    private final DataRepository repository = new DataRepository();
    private final DataRepository.Pending<Integer> pending = repository.register();

    @Actor
    public void complete() {
        repository.complete(pending.id(), 1);
    }

    @Actor
    public void close() {
        repository.close();
    }

    @Arbiter
    public void observe(II_Result result) {
        result.r1 = pending.future().isCompletedExceptionally() ? -1 : pending.future().join();
        result.r2 = repository.pendingCount();
    }
}
