package experiments.ordered.jcstress;

import com.lis.threadpool.OrderedThreadPoolExecutor;
import java.util.List;
import java.util.concurrent.AbstractExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
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
public class SequenceUniquenessStress {
    private final OrderedThreadPoolExecutor<Integer> executor =
            new OrderedThreadPoolExecutor<>(new LinkedBlockingQueue<>(), new InlineExecutor());

    @Actor
    public void actor1(JJ_Result result) {
        result.r1 = executor.executeOrdered(() -> 1);
    }

    @Actor
    public void actor2(JJ_Result result) {
        result.r2 = executor.executeOrdered(() -> 2);
    }

    @Arbiter
    public void cleanup() {
        executor.close();
    }

    private static final class InlineExecutor extends AbstractExecutorService {
        @Override public void shutdown() {}
        @Override public List<Runnable> shutdownNow() { return List.of(); }
        @Override public boolean isShutdown() { return false; }
        @Override public boolean isTerminated() { return false; }
        @Override public boolean awaitTermination(long timeout, TimeUnit unit) { return false; }
        @Override public void execute(Runnable command) { command.run(); }
    }
}
