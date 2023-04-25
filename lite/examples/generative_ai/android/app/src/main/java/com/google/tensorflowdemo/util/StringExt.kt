package com.google.tensorflowdemo.util

fun String.trimToMaxWordCount(count: Int): String {
    val allWords = allWords()
    val wordCount = allWords.size
    if (wordCount < count) return this

    val lastWords = allWords.toMutableList().subList(allWords.size - count, allWords.size)
    val lastText = lastWords.joinToString(separator = "")

    var inputIndex = this.length
    for (trimmedTextIndex in lastText.length - 1 downTo 0) {
        inputIndex--
        val trimmedChar = lastText[trimmedTextIndex]
        while (inputIndex >= 0 && this[inputIndex] != trimmedChar) {
            inputIndex--
        }
    }
    return this.substring(inputIndex)
}

fun String.splitToWords(): List<String> {
    val allWords = allWords()

    var index = 0
    val indexList = allWords.mapIndexed { wordIndex, word ->
        if (wordIndex == 0) {
            index += word.length
            0
        } else {
            val ch = word[0]
            while (index < length && this[index] != ch) index++
            val outputIndex = index
            index += word.length
            outputIndex
        }
    }

    return indexList.mapIndexed { i, wordStartIndex ->
        if (i < indexList.size - 1) {
            val wordEndIndex = indexList[i + 1]
            this.substring(wordStartIndex, wordEndIndex)
        } else this.substring(wordStartIndex)
    }
}

private val wordsRegex = """(\b\S+\b)""".toRegex()
fun String.allWords() = wordsRegex.findAll(this).toList().map { it.groupValues.first() }
