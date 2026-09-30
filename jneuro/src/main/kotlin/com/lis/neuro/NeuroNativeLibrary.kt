package com.lis.neuro

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.SymbolLookup
import java.lang.invoke.MethodHandle

internal object NeuroNativeLibrary {
    data class Loaded(val lookup: SymbolLookup, val name: String)

    fun open(arena: Arena, environment: String, candidates: List<String>, requiredSymbols: List<String>): Loaded {
        val requested = System.getenv(environment)?.trim().orEmpty()
        val names = buildList {
            if (requested.isNotEmpty()) add(requested)
            addAll(candidates)
        }.distinct()
        var lastFailure: Throwable? = null
        for (name in names) {
            try {
                val lookup = SymbolLookup.libraryLookup(name, arena)
                if (requiredSymbols.all { lookup.find(it).isPresent }) return Loaded(lookup, name)
            } catch (exception: RuntimeException) {
                lastFailure = exception
            } catch (exception: UnsatisfiedLinkError) {
                lastFailure = exception
            }
        }
        throw IllegalStateException("Unable to load " + environment + " from " + names.joinToString(), lastFailure)
    }

    fun downcall(loaded: Loaded, symbol: String, descriptor: FunctionDescriptor): MethodHandle =
        Linker.nativeLinker().downcallHandle(loaded.lookup.find(symbol).orElseThrow {
            IllegalStateException("Missing symbol " + symbol + " in " + loaded.name)
        }, descriptor)

    fun invokeInt(handle: MethodHandle, vararg arguments: Any?): Int =
        try {
            handle.invokeWithArguments(*arguments) as Int
        } catch (throwable: Throwable) {
            throw IllegalStateException("Native call failed", throwable)
        }
}
