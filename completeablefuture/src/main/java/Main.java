import java.util.concurrent.*;
import java.util.function.Function;

public class Main {

    private static final ExecutorService EXECUTOR = Executors.newFixedThreadPool(10);

    public static void main(String[] args) {
        Function<Integer,Integer> fn =(x) -> {
            System.out.println(Thread.currentThread().getName());
            System.out.println(x);
            return x+1;
        };
        CompletableFuture<Integer> f = CompletableFuture.completedFuture(0);
        f=f.thenApplyAsync(fn,EXECUTOR);
        f=f.thenApplyAsync(fn,EXECUTOR);
        f=f.thenApplyAsync(fn,EXECUTOR);
        f=f.thenApplyAsync(fn,EXECUTOR);
        f=f.thenApplyAsync(fn,EXECUTOR);
        f.join();
        EXECUTOR.shutdown();
    }
}
