package com.lis.neuro

import java.text.ParseException
import javax.swing.JFormattedTextField
import javax.swing.JSpinner
import javax.swing.SpinnerNumberModel
import javax.swing.text.DefaultFormatterFactory

internal object NumericInputs {
    fun spinner(value: Number, step: Number): JSpinner {
        val model = object : SpinnerNumberModel(value, null, null, step) {
            override fun getNextValue(): Any? = adjacent(1)
            override fun getPreviousValue(): Any? = adjacent(-1)
            private fun adjacent(direction: Int): Number? {
                val current = number
                return when (current) {
                    is Int -> (current.toLong() + direction * stepSize.toLong()).takeIf { it in Int.MIN_VALUE..Int.MAX_VALUE }?.toInt()
                    is Long -> try { Math.addExact(current, Math.multiplyExact(direction.toLong(), stepSize.toLong())) } catch (_: ArithmeticException) { null }
                    else -> (current.toDouble() + direction * stepSize.toDouble()).takeIf { it.isFinite() }
                }
            }
        }
        return JSpinner(model).apply {
            val field = (editor as JSpinner.DefaultEditor).textField
            field.formatterFactory = DefaultFormatterFactory(object : JFormattedTextField.AbstractFormatter() {
                override fun stringToValue(text: String): Any = parse(text, value)
                override fun valueToString(value: Any?): String = value?.toString() ?: ""
            })
            field.value = value
        }
    }

    internal fun parse(text: String, type: Number): Number {
        val value = text.trim()
        val parsed = when (type) {
            is Int -> value.toIntOrNull()
            is Long -> value.toLongOrNull()
            else -> value.toDoubleOrNull()?.takeIf { it.isFinite() }
        }
        return parsed ?: throw ParseException("Enter a finite ${if (type is Int || type is Long) "integer representable by ${type.javaClass.simpleName}" else "number"}.", 0)
    }

    fun parameterCount(topology: IntArray): Int {
        require(topology.size >= 2 && topology.all { it > 0 }) { "Topology requires positive input and output widths." }
        var total = 0L
        for (layer in 1 until topology.size) {
            total += (topology[layer - 1].toLong() + 1) * topology[layer]
            require(total <= Int.MAX_VALUE) { "Topology exceeds the JVM array/index representation." }
        }
        return total.toInt()
    }

    fun checkAllocation(topology: IntArray) {
        val parameters = parameterCount(topology)
        val runtime = Runtime.getRuntime()
        val available = runtime.maxMemory() - runtime.totalMemory() + runtime.freeMemory()
        val workingBytes = parameters.toLong() * 64 + topology.sumOf { it.toLong() } * 64
        require(workingBytes <= available / 2) { "Not enough JVM heap for this topology. Increase -Xmx or choose a smaller network." }
    }
}
