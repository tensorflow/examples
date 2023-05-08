package com.google.tensorflowdemo.util

import org.junit.Assert.*
import org.junit.Test

class StringExtKtTest {
    @Test fun TestThatShortTextIsLeftUnharmed() {
        val input = "Mary had a little lamb!"
        val output = input.trimToMaxWordCount(20)
        assertEquals(input, output)
    }

    @Test fun TestThatATextWithMoreWordsThanAllowedIsTrimmedProperly() {
        val input = "Mary had a little, lamb"
        val output = input.trimToMaxWordCount(2)
        assertTrue(output.length < input.length)
    }

    @Test fun TestThatAnEmptyTextIsTrimmedProperly() {
        val input = ""
        val output = input.trimToMaxWordCount(5)
        assertTrue(output.isEmpty())
    }

    @Test fun TestThatASentenceIsSplitIntoWordsProperly() {
        val input = "Mary had a little lamb!"
        val output = input.splitToWords()
        assertEquals(5, output.size)
    }

    @Test fun TestThatAnEmptyStringIsSplitIntoWordsProperly() {
        val input = ""
        val output = input.splitToWords()
        assertEquals(0, output.size)
    }

    @Test fun TestThatNonWordsAtStartAreKeptAsIs() {
        val input = ". A few of us, a couple of guys"
        val output = input.splitToWords()
        assertEquals(8, output.size)
        assertTrue(output.first().startsWith(input.substring(0, 2)))
    }
}