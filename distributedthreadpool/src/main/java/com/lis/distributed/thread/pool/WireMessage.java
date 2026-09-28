package com.lis.distributed.thread.pool;

import com.lis.distributed.thread.pool.func.SerializableSupplier;

import java.io.PrintWriter;
import java.io.Serializable;
import java.io.StringWriter;
import java.util.Objects;

sealed interface WireMessage extends Serializable
        permits WireMessage.Invocation, WireMessage.Command, WireMessage.Response, WireMessage.Registration {

    record Invocation(long requestId, SerializableSupplier<?> task) implements WireMessage {
        public Invocation {
            if (requestId < 0) {
                throw new IllegalArgumentException("requestId must be non-negative");
            }
            Objects.requireNonNull(task, "task");
        }
    }

    record Command(long requestId, TaskMessage<?> task) implements WireMessage {
        public Command {
            if (requestId < 0) {
                throw new IllegalArgumentException("requestId must be non-negative");
            }
            Objects.requireNonNull(task, "task");
        }
    }

    record Response(long requestId, Object value, RemoteFailure failure) implements WireMessage {
        public Response {
            if (requestId < 0) {
                throw new IllegalArgumentException("requestId must be non-negative");
            }
            if (failure == null && value != null && !(value instanceof Serializable)) {
                throw new IllegalArgumentException("response value must be Serializable");
            }
        }

        static Response success(long requestId, Object value) {
            return new Response(requestId, value, null);
        }

        static Response failure(long requestId, Throwable failure) {
            return new Response(requestId, null, RemoteFailure.from(failure));
        }
    }

    record Registration(int clientId) implements WireMessage {
        public Registration {
            if (clientId <= 0) {
                throw new IllegalArgumentException("clientId must be positive");
            }
        }
    }

    record RemoteFailure(String type, String message, String stackTrace) implements Serializable {
        public RemoteFailure {
            Objects.requireNonNull(type, "type");
            Objects.requireNonNull(stackTrace, "stackTrace");
        }

        static RemoteFailure from(Throwable failure) {
            Objects.requireNonNull(failure, "failure");
            var writer = new StringWriter();
            failure.printStackTrace(new PrintWriter(writer));
            return new RemoteFailure(failure.getClass().getName(), failure.getMessage(), writer.toString());
        }

        RemoteExecutionException toException() {
            return new RemoteExecutionException(type, message, stackTrace);
        }
    }
}
