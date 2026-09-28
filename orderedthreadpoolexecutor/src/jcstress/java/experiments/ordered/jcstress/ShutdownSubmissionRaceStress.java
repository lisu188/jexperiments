package experiments.ordered.jcstress;

import com.lis.threadpool.OrderedThreadPoolExecutor;
import java.util.List;
import java.util.concurrent.AbstractExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import org.openjdk.jcstress.annotations.Actor;
import org.openjdk.jcstress.annotations.Expect;
import org.openjdk.jcstress.annotations.JCStressTest;
import org.openjdk.jcstress.annotations.Outcome;
import org.openjdk.jcstress.annotations.State;
import org.openjdk.jcstress.infra.results.II_Result;

@JCStressTest
@Outcome(id = {"1, 1", "0, 1"}, expect = Expect.ACCEPTABLE)
@State
public class ShutdownSubmissionRaceStress {
    private final OrderedThreadPoolExecutor<Integer> executor =
            new OrderedThreadPoolExecutor<>(new LinkedBlockingQueue<>(), new InlineExecutor());

    @Actor
    public void submit(II_Result result) {
        try {
            executor.executeOrdered(() -> 1);
            result.r1 = 1;
        } catch (RejectedExecutionException expected) {
            result.r1 = 0;
        }
    }

    @Actor
    public void shutdown(II_Result result) {
        executor.shutdown();
        result.r2 = executor.isShutdown() ? 1 : 0;
    }

    private static final class InlineExecutor extends AbstractExecutorService {
        private volatile boolean shutdown;

        @Override public void shutdown() { shutdown = true; }
        @Override public List<Runnable> shutdownNow() { shutdown = true; return List.of(); }
        @Override public boolean isShutdown() { return shutdown; }
        @Override public boolean isTerminated() { return shutdown; }
        @Override public boolean awaitTermination(long timeout, TimeUnit unit) { return shutdown; }
        @Override public void execute(Runnable command) {
            if (shutdown) {
                throw new RejectedExecutionException();
            }
            command.run();
        }
    }
}
