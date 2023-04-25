package com.google.tensorflowdemo.util

import org.junit.Assert.*
import org.junit.Test

class StringExtKtTest {
    @Test
    fun `Test that short text is left unharmed`() {
        val input = "Mary had a little lamb!"
        val output = input.trimToMaxWordCount(20)
        assertEquals(input, output)
    }

    @Test
    fun `Test that a text with more words than allowed is trimmed properly`() {
        val input = "Mary had a little, lamb"
        val output = input.trimToMaxWordCount(2)
        assertTrue(output.length < input.length)
    }

    @Test
    fun `Test that an empty text is trimmed properly`() {
        val input = ""
        val output = input.trimToMaxWordCount(5)
        assertTrue(output.isEmpty())
    }

    @Test
    fun `Test that a sentence is split into words properly`() {
        val input = "Mary had a little lamb!"
        val output = input.splitToWords()
        assertEquals(5, output.size)
    }

    @Test
    fun `Test that an empty string is split into words properly`() {
        val input = ""
        val output = input.splitToWords()
        assertEquals(0, output.size)
    }

    @Test
    fun `Test that non-words at start are kept as is`() {
        val input = ". A few of us, a couple of guys"
        val output = input.splitToWords()
        assertEquals(8, output.size)
        assertTrue(output.first().startsWith(input.substring(0, 2)))
    }
}