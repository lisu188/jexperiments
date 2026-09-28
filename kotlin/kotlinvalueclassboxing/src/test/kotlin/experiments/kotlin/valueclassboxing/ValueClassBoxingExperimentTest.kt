package experiments.kotlin.valueclassboxing

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Test

class ValueClassBoxingExperimentTest {
    @Test
    fun coversValueClassAndNullableBoxingPaths() {
        val id = CustomerId("T-100")
        val registry = CustomerRegistry()

        assertEquals("customer:T-100", id.display())
        assertEquals("CustomerId(T-100)", id.toString())
        assertEquals("direct=T-100", registry.accept(id))
        assertEquals("nullable=T-100", registry.acceptNullable(id))
        assertEquals("nullable=missing", registry.acceptNullable(null))
        assertEquals("T-100|T-200", registry.collect(listOf(id, CustomerId("T-200"))))
        assertEquals("CustomerId|raw=J-400", JavaValueClassCaller.reflectCustomerId("J-400"))

        val constructor = JavaValueClassCaller::class.java.getDeclaredConstructor()
        constructor.isAccessible = true
        assertNotNull(constructor.newInstance())
    }
}
