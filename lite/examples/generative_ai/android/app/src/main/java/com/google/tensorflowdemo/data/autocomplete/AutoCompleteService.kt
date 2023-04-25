package com.google.tensorflowdemo.data.autocomplete

import android.content.Context
import com.google.tensorflowdemo.util.splitToWords
import com.google.tensorflowdemo.util.trimToMaxWordCount
import com.mediamonks.wordfilter.LanguageChecker
import io.github.aakira.napier.Napier
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.min

/**
 * Possible errors from [AutoCompleteService]
 */
enum class AutoCompleteServiceError {
    MODEL_FILE_NOT_FOUND,
    MODEL_NOT_INITIALIZED,
    NO_SUGGESTIONS,
    BAD_LANGUAGE,
}

/**
 * Result from [AutoCompleteService.getSuggestion] method call
 */
sealed interface AutoCompleteResult {
    data class Success(val words: List<String>) : AutoCompleteResult
    data class Error(val error: AutoCompleteServiceError) : AutoCompleteResult
}

/**
 * Result from [AutoCompleteService.initModel] method call
 */
sealed interface InitModelResult {
    object Success : InitModelResult
    data class Error(val error: AutoCompleteServiceError) : InitModelResult
}

interface AutoCompleteService {

    /**
     * Configuration setting boundaries for model input
     */
    val inputConfiguration: AutoCompleteInputConfiguration

    /**
     * Boolean indicating whether the model has been initialized successfully
     */
    val isInitialized: Boolean

    /**
     * Initialize TensorFlow-Lite with app provided model
     * @return [InitModelResult.Success] if TFLite was initialized properly, otherwise [InitModelResult.Error]
     */
    suspend fun initModel(): InitModelResult

    /**
     * Get autocomplete suggestion split into words for the provided [input].
     * If [applyWindow] is true, the last [windowSize] words are taken from [input] and fed into the interpreter.
     * @return an instance of [AutoCompleteResult.Error] if something went wrong, or
     * an instance of [AutoCompleteResult.Success] with the suggested text, split into words
     */
    suspend fun getSuggestion(input: String, applyWindow: Boolean = false, windowSize: Int = 50): AutoCompleteResult
}

class AutoCompleteServiceImpl(
    private val context: Context,
    private val languageChecker: LanguageChecker,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) : AutoCompleteService, AutoCloseable {

    override val inputConfiguration = AutoCompleteInputConfiguration(
        minWordCount = 5,
        maxWordCount = min(50, MAX_INPUT_WORD_COUNT),
        initialWordCount = 20
    )

    override var isInitialized: Boolean = false
        private set

    private lateinit var interpreter: Interpreter
    private val outputBuffer = ByteBuffer.allocateDirect(OUTPUT_BUFFER_SIZE)

    override suspend fun initModel(): InitModelResult {
        return withContext(dispatcher) {
            val loadResult = loadModelFile(context)

            if (loadResult.isFailure) {
                val exc = loadResult.exceptionOrNull()
                return@withContext if (exc is FileNotFoundException) {
                    InitModelResult.Error(AutoCompleteServiceError.MODEL_FILE_NOT_FOUND)
                } else {
                    InitModelResult.Error(AutoCompleteServiceError.MODEL_NOT_INITIALIZED)
                }
            }

            val model = loadResult.getOrNull()
            isInitialized = model?.let {
                interpreter = Interpreter(it)
                true
            } ?: false

            if (isInitialized) InitModelResult.Success
            else InitModelResult.Error(AutoCompleteServiceError.MODEL_NOT_INITIALIZED)
        }
    }

    override suspend fun getSuggestion(input: String, applyWindow: Boolean, windowSize: Int) = withContext(dispatcher) {
        Napier.d { "[0] Start interpretation" }
        Napier.d { "[1] Input text: (${input.length} chars) '$input'" }

        if (languageChecker.containsBadLanguage(input)) {
            Napier.w { "[2] Input contains bad language, refused!" }
            return@withContext AutoCompleteResult.Error(AutoCompleteServiceError.BAD_LANGUAGE)
        }

        if (!::interpreter.isInitialized) {
            val result = initModel()
            if (result is InitModelResult.Error) {
                return@withContext AutoCompleteResult.Error(result.error)
            }
        }

        val maxInputWordCount = if (applyWindow) windowSize else MAX_INPUT_WORD_COUNT
        Napier.d { "[2] Trimming input to max $maxInputWordCount words" }

        val trimmedInput = input.trimToMaxWordCount(maxInputWordCount)
        Napier.d { "[3] Model input: (${trimmedInput.length} chars) '$trimmedInput'" }

        var retryCount = 0
        var containsBadLanguage: Boolean
        lateinit var output:String
        do {
            output = runInterpreterOn(trimmedInput)

            containsBadLanguage = languageChecker.containsBadLanguage(output)

            retryCount++
        } while (containsBadLanguage && retryCount < RETRY_COUNT_ON_BAD_LANGUAGE)

        if (containsBadLanguage) {
            Napier.w { "[4] Output still contains bad language after 3 attempts, refused!" }

            return@withContext AutoCompleteResult.Error(AutoCompleteServiceError.NO_SUGGESTIONS)
        }

        Napier.d { "[4] Model output: (${output.length} chars) '$output'" }

        if (output.length < trimmedInput.length) {
            Napier.w { "[5] NO SUGGESTION: Output length is shorter than trimmed input length, so there was no new text suggested" }
            AutoCompleteResult.Error(AutoCompleteServiceError.NO_SUGGESTIONS)
        } else {
            val newText = output.substring(trimmedInput.length)
            Napier.d { "[5] New text from interpreter: (${newText.length} chars) '$newText'" }

            val words = newText.splitToWords()
            if (words.isEmpty()) {
                Napier.w { "[6] NO SUGGESTION: No words found after splitting new text into words" }
                AutoCompleteResult.Error(AutoCompleteServiceError.NO_SUGGESTIONS)
            } else {
                Napier.d { "[6] New text split in words: (${words.size} words) [${words.joinToString()}]" }
                AutoCompleteResult.Success(words)
            }
        }
    }

    private fun runInterpreterOn(input: String): String {
        outputBuffer.clear()

        interpreter.run(input, outputBuffer)

        outputBuffer.flip().position(12)
        val bytes = ByteArray(outputBuffer.remaining())
        outputBuffer.get(bytes)
        outputBuffer.clear()

        return String(bytes, Charsets.UTF_8)
    }

    private fun loadModelFile(context: Context): Result<MappedByteBuffer?> {
        try {
            val descriptor = context.assets.openFd(TFLITE_MODEL)

            FileInputStream(descriptor.fileDescriptor).use { stream ->
                return Result.success(
                    stream.channel.map(
                        /* mode = */ FileChannel.MapMode.READ_ONLY,
                        /* position = */ descriptor.startOffset,
                        /* size = */ descriptor.declaredLength
                    )
                )
            }
        } catch (e: Exception) {
            Napier.e { "Failed to load model: ${e.localizedMessage}" }

            return Result.failure(e)
        }
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        private const val TFLITE_MODEL = "autocomplete.tflite"
        private const val OUTPUT_BUFFER_SIZE = 800
        private const val MAX_INPUT_WORD_COUNT = 1024
        private const val RETRY_COUNT_ON_BAD_LANGUAGE = 3
    }
}

data class AutoCompleteInputConfiguration(
    val minWordCount: Int,
    val maxWordCount: Int,
    val initialWordCount: Int
)