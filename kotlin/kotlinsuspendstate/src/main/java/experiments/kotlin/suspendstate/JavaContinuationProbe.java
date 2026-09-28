package experiments.kotlin.suspendstate;

import kotlin.coroutines.Continuation;
import kotlin.coroutines.CoroutineContext;
import kotlin.coroutines.EmptyCoroutineContext;
import kotlin.coroutines.intrinsics.IntrinsicsKt;

public final class JavaContinuationProbe {
    private JavaContinuationProbe() {
    }

    public static Object callDelayedDouble(int seed) {
        return SuspendStateExperimentKt.delayedDouble(seed, new Continuation<String>() {
            @Override
            public CoroutineContext getContext() {
                return EmptyCoroutineContext.INSTANCE;
            }

            @Override
            public void resumeWith(Object result) {
                System.out.println("java-continuation-result=" + result);
            }
        });
    }

    public static String describeReturn(Object returned) {
        if (returned == IntrinsicsKt.getCOROUTINE_SUSPENDED()) {
            return "COROUTINE_SUSPENDED";
        }
        return String.valueOf(returned);
    }
}
