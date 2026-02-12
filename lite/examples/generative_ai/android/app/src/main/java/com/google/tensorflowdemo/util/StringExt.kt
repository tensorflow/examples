/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
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
