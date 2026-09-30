package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.invoke.MethodHandles
import java.lang.invoke.MethodType
import java.util.Optional
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class NativeSmallTrainingTest {
    private val hp = Neuro.HyperParameters(0.11, 0.31, 0.75, 42)
    private fun state(samples: Int = 19) = preparedSmall(intArrayOf(2, 8, 8, 8, 1), hp, samples).exportTrainingState()

    @Test fun oneFfmCallPerChunkMarshalsIndependentOrdersAndPersistsAllNativeBuffers() {
        val original = state()
        val fake = NativeSmallFake()
        val states = Array(5) { original }
        val orders = Array(5) { model -> Array(3) { epoch -> smallOrders(model + epoch + 1, 19).last() } }
        NativeSmallCohort(states, hp.copy(sigmoidMode = Neuro.SigmoidMode.FAST), loader = fake.loader).use { kernel ->
            assertEquals(256, kernel.info.simdBits)
            assertEquals("FP64", kernel.info.precision)
            assertEquals(TrainingEngine.SMALL, kernel.info.engine)
            assertEquals("FAST", kernel.info.sigmoid)
            val first = kernel.train(orders, 7, false)
            states.indices.forEach { assertSmallState(states[it], first[it], 0.0) }
            assertEquals(1, fake.calls.size)
            assertEquals(listOf(5, 19, 3, 7, 0, 0.11, 0.31, 0.75, 1, 1), fake.calls[0].drop(4))
            assertArrayEquals(orders.flatMap { model -> model.flatMap { it.toList() } }.toIntArray(), fake.lastOrders)
            kernel.train(Array(5) { smallOrders(64, 19) }, 99, true)
            assertEquals(2, fake.calls.size)
            assertEquals(1, fake.calls[1][7])
            assertEquals(1, fake.calls[1][8])
            for (pointer in 0..3) assertEquals((fake.calls[0][pointer] as MemorySegment).address(),
                (fake.calls[1][pointer] as MemorySegment).address())
            first[0].weights[0][0] = 99.0
            assertSmallState(original, kernel.train(orders, 7, false)[0], 0.0)
        }
        assertFalse(fake.arena!!.scope().isAlive)
        fake.assertHealthy()
    }

    @Test fun nativeOutputIsDecodedAsParameterAndMomentumSoaWithoutChangingInputState() {
        val initial = state()
        val fake = NativeSmallFake().apply { mutate = true }
        NativeSmallCohort(Array(4) { initial }, hp, loader = fake.loader).use { kernel ->
            val output = kernel.train(Array(4) { smallOrders(1, 19) }, 1, true)
            output.indices.forEach { model ->
                assertEquals(initial.weights[0][0] + model + 1.0, output[model].weights[0][0])
                assertEquals(initial.biasVelocity.last()[0] + model + 1.0, output[model].biasVelocity.last()[0])
            }
            assertArrayEquals(initial.weights[0], state().weights[0])
        }
        fake.assertHealthy()
    }

    @Test fun runtimeCapabilityAndExplicitScalarChoiceAreTruthful() {
        for ((capable, allow, scalar, count) in listOf(listOf(0, 1, 0, 4), listOf(1, 0, 0, 4),
            listOf(1, 1, 1, 4), listOf(1, 1, 0, 1))) {
            val fake = NativeSmallFake().apply { avx2 = capable }
            NativeSmallCohort(Array(count) { state() }, if (scalar == 1) hp.copy(kernel = Neuro.Kernel.SCALAR) else hp,
                allowSimd = allow == 1, loader = fake.loader).use { kernel ->
                assertEquals(0, kernel.info.simdBits)
                kernel.train(Array(count) { smallOrders(1, 19) }, 1, false)
                assertEquals(0, fake.calls.single().last())
            }
            fake.assertHealthy()
        }
    }

    @Test fun nativeErrorsAndNonfiniteOutputAreTerminalAndDoNotPublish() {
        for (nonfinite in listOf(false, true)) {
            val initial = state()
            val fake = NativeSmallFake().apply { if (nonfinite) nan = true else status = 3 }
            val kernel = NativeSmallTraining(initial, hp, loader = fake.loader)
            assertThrows(IllegalStateException::class.java) { kernel.train(smallOrders(2, 19), 7, false) }
            assertThrows(IllegalStateException::class.java) { kernel.train(smallOrders(1, 19), 7, false) }
            assertEquals(1, fake.calls.size)
            assertSmallState(state(), initial, 0.0)
            kernel.close(); kernel.close()
            assertFalse(fake.arena!!.scope().isAlive)
            fake.assertHealthy()
        }
    }

    @Test fun inputValidationDoesNotPoisonTheKernelAndEmptyDatasetsAvoidNativeCall() {
        val fake = NativeSmallFake()
        NativeSmallCohort(arrayOf(state()), hp, loader = fake.loader).use { kernel ->
            val valid = arrayOf(smallOrders(1, 19))
            assertThrows(IllegalArgumentException::class.java) { kernel.train(valid, 0, false) }
            assertThrows(IllegalArgumentException::class.java) { kernel.train(emptyArray(), 1, false) }
            assertThrows(IllegalArgumentException::class.java) { kernel.train(arrayOf(emptyArray()), 1, false) }
            assertThrows(IllegalArgumentException::class.java) { kernel.train(arrayOf(smallOrders(65, 19)), 1, false) }
            assertThrows(IllegalArgumentException::class.java) { kernel.train(arrayOf(arrayOf(IntArray(19))), 1, false) }
            kernel.train(valid, 1, false)
            assertEquals(1, fake.calls.size)
        }
        val empty = NativeSmallFake()
        val kernel = NativeSmallTraining(state(0), hp, loader = empty.loader)
        assertSmallState(state(0), kernel.train(arrayOf(IntArray(0)), 1, true), 0.0)
        assertEquals(0, kernel.info.simdBits)
        assertTrue(empty.calls.isEmpty())
        kernel.close()
        assertThrows(IllegalStateException::class.java) { kernel.train(arrayOf(IntArray(0)), 1, true) }
        fake.assertHealthy(); empty.assertHealthy()
    }

    @Test fun constructorsRejectUnsupportedScopeAndReleaseArenaWhenLoadingFails() {
        val fake = NativeSmallFake()
        assertThrows(IllegalArgumentException::class.java) { NativeSmallCohort(emptyArray(), hp, loader = fake.loader) }
        assertThrows(IllegalArgumentException::class.java) { NativeSmallTraining(state(), hp, Neuro.TrainingPrecision.FP32, fake.loader) }
        assertThrows(IllegalArgumentException::class.java) {
            NativeSmallTraining(preparedSmall(intArrayOf(2, 4, 1), hp).exportTrainingState(), hp, loader = fake.loader)
        }
        assertThrows(IllegalArgumentException::class.java) { NativeSmallCohort(arrayOf(state(), state(20)), hp, loader = fake.loader) }
        var captured: Arena? = null
        assertThrows(UnsatisfiedLinkError::class.java) {
            NativeSmallTraining(state(), hp, loader = { captured = it; throw UnsatisfiedLinkError("missing test library") })
        }
        assertFalse(captured!!.scope().isAlive)
        fake.abi = 2
        assertThrows(IllegalArgumentException::class.java) { NativeSmallTraining(state(), hp, loader = fake.loader) }
        assertFalse(fake.arena!!.scope().isAlive)
        fake.abi = 1
        fake.missingTrain = true
        assertThrows(IllegalStateException::class.java) { NativeSmallTraining(state(), hp, loader = fake.loader) }
        assertFalse(fake.arena!!.scope().isAlive)
    }
}

private class NativeSmallFake {
    var abi = 1
    var avx2 = 1
    var status = 0
    var nan = false
    var mutate = false
    var missingTrain = false
    var arena: Arena? = null
    var lastOrders = IntArray(0)
    val calls = ArrayList<List<Any>>()
    private val failures = ArrayList<Throwable>()
    val loader: (Arena) -> NeuroNativeLibrary.Loaded = { memory ->
        arena = memory
        val descriptors = mapOf("jsmall_abi" to FunctionDescriptor.of(JAVA_INT),
            "jsmall_avx2" to FunctionDescriptor.of(JAVA_INT), "jsmall_train" to NativeSmallCohort.TRAIN_DESCRIPTOR)
        val symbols = descriptors.filterKeys { !missingTrain || it != "jsmall_train" }.mapValues { (name, descriptor) ->
            val callback = Callback { args ->
                try {
                    when (name) {
                        "jsmall_abi" -> abi
                        "jsmall_avx2" -> avx2
                        else -> {
                            calls += args.toList()
                            val models = args[4] as Int
                            val count = models * (args[5] as Int) * (args[6] as Int)
                            lastOrders = (args[3] as MemorySegment).reinterpret(count * 4L).toArray(JAVA_INT)
                            val state = (args[0] as MemorySegment).reinterpret(models * 354L * 8)
                            if (nan) state.set(JAVA_DOUBLE, 0, Double.NaN)
                            if (mutate) for (index in 0 until models * 354) state.setAtIndex(JAVA_DOUBLE, index.toLong(),
                                state.getAtIndex(JAVA_DOUBLE, index.toLong()) + index % models + 1.0)
                            status
                        }
                    }
                } catch (failure: Throwable) { failures += failure; 9999 }
            }
            val handle = MethodHandles.lookup().findVirtual(Callback::class.java, "invoke",
                MethodType.methodType(Int::class.javaPrimitiveType, Array<Any>::class.java)).bindTo(callback)
                .asCollector(Array<Any>::class.java, descriptor.argumentLayouts().size).asType(descriptor.toMethodType())
            Linker.nativeLinker().upcallStub(handle, descriptor, memory)
        }
        NeuroNativeLibrary.Loaded(SymbolLookup { Optional.ofNullable(symbols[it]) }, "test-native-small")
    }
    fun assertHealthy() = assertTrue(failures.isEmpty(), failures.joinToString())
    private class Callback(val action: (Array<Any>) -> Int) { fun invoke(arguments: Array<Any>): Int = action(arguments) }
}
