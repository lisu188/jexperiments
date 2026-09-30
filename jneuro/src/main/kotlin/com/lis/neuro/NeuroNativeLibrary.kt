package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.SymbolLookup
import java.lang.invoke.MethodHandle
import java.util.concurrent.ConcurrentHashMap

internal typealias NeuroLibraryLoader = (Arena, String, List<String>, List<String>) -> NeuroNativeLibrary.Loaded

internal object NeuroNativeLibrary {
    data class Loaded(val lookup: SymbolLookup, val name: String)
    private val reportedLibraries = ConcurrentHashMap.newKeySet<String>()

    fun open(arena: Arena, environment: String, candidates: List<String>, requiredSymbols: List<String>): Loaded =
        open(arena, environment, candidates, requiredSymbols, System::getenv, SymbolLookup::libraryLookup)

    internal fun open(arena: Arena, environment: String, candidates: List<String>, requiredSymbols: List<String>,
                      environmentValue: (String) -> String?, lookupLibrary: (String, Arena) -> SymbolLookup): Loaded {
        val requested = environmentValue(environment)?.trim().orEmpty()
        val names = buildList {
            if (requested.isNotEmpty()) add(requested)
            addAll(candidates)
        }.distinct()
        var lastFailure: Throwable? = null
        for (name in names) {
            NeuroLog.debug("native", "library.attempt") { mapOf("setting" to environment, "library" to name) }
            try {
                val lookup = lookupLibrary(name, arena)
                val missing = requiredSymbols.filterNot { lookup.find(it).isPresent }
                if (missing.isEmpty()) {
                    selected(environment, name)
                    return Loaded(lookup, name)
                }
                NeuroLog.debug("native", "library.incomplete") {
                    mapOf("setting" to environment, "library" to name, "missingSymbols" to missing.joinToString(","))
                }
            } catch (exception: RuntimeException) {
                lastFailure = exception
                NeuroLog.debug("native", "library.rejected") { mapOf("library" to name, "reason" to exception.javaClass.simpleName) }
            } catch (exception: UnsatisfiedLinkError) {
                lastFailure = exception
                NeuroLog.debug("native", "library.rejected") { mapOf("library" to name, "reason" to exception.javaClass.simpleName) }
            }
        }
        NeuroLog.debug("native", "library.unavailable") { mapOf("setting" to environment, "candidates" to names.joinToString(",")) }
        throw IllegalStateException("Unable to load " + environment + " from " + names.joinToString(), lastFailure)
    }

    internal fun selected(setting: String, name: String) {
        if (reportedLibraries.add("$setting:$name")) NeuroLog.info("native", "library.selected", "setting" to setting, "library" to name)
    }

    internal inline fun checkStatus(status: Int, operation: String, component: String, message: () -> String) {
        if (status == 0) return
        val failure = IllegalStateException(message())
        NeuroLog.error(component, "native.failed", failure, "operation" to operation, "status" to status,
            "hint" to "Check CUDA library and driver compatibility, device access, and available GPU memory.")
        throw failure
    }

    fun downcall(loaded: Loaded, symbol: String, descriptor: FunctionDescriptor): MethodHandle =
        Linker.nativeLinker().downcallHandle(loaded.lookup.find(symbol).orElseThrow {
            IllegalStateException("Missing symbol " + symbol + " in " + loaded.name)
        }, descriptor)

    fun invokeInt(handle: MethodHandle, vararg arguments: Any?): Int =
        try {
            handle.invokeWithArguments(*arguments) as Int
        } catch (throwable: Throwable) {
            NeuroLog.error("native", "native.invocation.failed", throwable)
            throw IllegalStateException("Native call failed", throwable)
        }
}
