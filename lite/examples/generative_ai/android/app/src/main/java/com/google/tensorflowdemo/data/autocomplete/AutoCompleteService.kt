package com.google.tensorflowdemo.data.autocomplete

import android.content.Context
import androidx.annotation.WorkerThread
import com.google.tensorflowdemo.data.autocomplete.AutoCompleteService.AutoCompleteInputConfiguration
import com.google.tensorflowdemo.data.autocomplete.AutoCompleteService.AutoCompleteResult
import com.google.tensorflowdemo.data.autocomplete.AutoCompleteService.AutoCompleteServiceError
import com.google.tensorflowdemo.data.autocomplete.AutoCompleteService.InitModelResult
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

interface AutoCompleteService {

    /**
     * Configuration of input for model
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

    data class AutoCompleteInputConfiguration(
        // Minimum number of words to be taken from the end of the input text
        val minWordCount: Int = 5,
        // Maximum number of words to be taken from the end of the input text
        val maxWordCount: Int = 50,
        // Initially selected value for number of words to be taken from the end of the input text
        val initialWordCount: Int = 20,
    )
}

class AutoCompleteServiceImpl(
    private val context: Context,
    private val languageChecker: LanguageChecker,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) : AutoCompleteService, AutoCloseable {

    /**
     * The values below are used to configure the slider that allows a user to select the number of words to take from the current text
     * and use as input for the model to generate new text from.
     */
    override val inputConfiguration = AutoCompleteInputConfiguration(
        // Minimum number of words to be taken from the end of the input text
        minWordCount = 5,
        // Maximum number of words to be taken from the end of the input text, limited by what the model allows
        maxWordCount = min(50, MAX_INPUT_WORD_COUNT),
        // Initially selected value for number of words to be taken from the end of the input text
        initialWordCount = 20
    )

    override var isInitialized: Boolean = false
        private set

    private lateinit var interpreter: Interpreter
    private val outputBuffer = ByteBuffer.allocateDirect(OUTPUT_BUFFER_SIZE)


    /**
     * Initialize TensorFlow Lite with app provided model
     * @return [InitModelResult.Success] if TFLite was initialized properly, otherwise [InitModelResult.Error]
     */
    override suspend fun initModel(): InitModelResult {
        return withContext(dispatcher) {
            // Load model file
            val loadResult = loadModelFile(context)

            // Determine if load was successful
            if (loadResult.isFailure) {
                val exc = loadResult.exceptionOrNull()
                return@withContext if (exc is FileNotFoundException) {
                    InitModelResult.Error(AutoCompleteServiceError.MODEL_FILE_NOT_FOUND)
                } else {
                    InitModelResult.Error(AutoCompleteServiceError.MODEL_NOT_INITIALIZED)
                }
            }

            // Instantiate interpreter with loaded model
            val model = loadResult.getOrNull()
            isInitialized = model?.let {
                interpreter = Interpreter(it)
                true
            } ?: false

            if (isInitialized) InitModelResult.Success
            else InitModelResult.Error(AutoCompleteServiceError.MODEL_NOT_INITIALIZED)
        }
    }

    /**
     * Get autocomplete suggestion split into words for the provided [input].
     * If [applyWindow] is true, the last [windowSize] words are taken from [input] and fed into the interpreter.
     * @return an instance of [AutoCompleteResult.Error] if something went wrong, or
     * an instance of [AutoCompleteResult.Success] with the suggested text, split into words
     */
    override suspend fun getSuggestion(input: String, applyWindow: Boolean, windowSize: Int) = withContext(dispatcher) {
        Napier.d { "[0] Start interpretation" }
        Napier.d { "[1] Input text: (${input.length} chars) '$input'" }

        // Check input for bad language
        if (languageChecker.containsBadLanguage(input)) {
            Napier.w { "[2] Input contains bad language, refused!" }
            return@withContext AutoCompleteResult.Error(AutoCompleteServiceError.BAD_LANGUAGE)
        }

        // Initialize interpreter if necessary
        if (!::interpreter.isInitialized) {
            val result = initModel()
            if (result is InitModelResult.Error) {
                return@withContext AutoCompleteResult.Error(result.error)
            }
        }

        // Determine maximum number of words to take from input as model input
        val maxInputWordCount = if (applyWindow) windowSize else MAX_INPUT_WORD_COUNT
        Napier.d { "[2] Trimming input to max $maxInputWordCount words" }

        // Trim input text to  maximum number of words
        val trimmedInput = input.trimToMaxWordCount(maxInputWordCount)
        Napier.d { "[3] Model input: (${trimmedInput.length} chars) '$trimmedInput'" }

        var retryCount = 0
        var containsBadLanguage: Boolean
        lateinit var output: String

        // Run generation until it no longer contains bad language or a max number of tries has been exceeded
        do {
            // Let model generate new text based on windowed input
            output = runInterpreterOn(trimmedInput)

            // Check output for bad language
            containsBadLanguage = languageChecker.containsBadLanguage(output)

            retryCount++
        } while (containsBadLanguage && retryCount < RETRY_COUNT_ON_BAD_LANGUAGE)

        // Return error if output still contains bad language
        if (containsBadLanguage) {
            Napier.w { "[4] Output still contains bad language after 3 attempts, refused!" }

            return@withContext AutoCompleteResult.Error(AutoCompleteServiceError.NO_SUGGESTIONS)
        }

        Napier.d { "[4] Model output: (${output.length} chars) '$output'" }

        // Check if output size is actually longer than original input text, if not that's an error
        if (output.length < trimmedInput.length) {
            Napier.w { "[5] NO SUGGESTION: Output length is shorter than trimmed input length, so there was no new text suggested" }
            AutoCompleteResult.Error(AutoCompleteServiceError.NO_SUGGESTIONS)
        } else {
            // Output = input + new text, determine new text by subtracting input
            val newText = output.substring(output.indexOf(trimmedInput) + trimmedInput.length)
            Napier.d { "[5] New text from interpreter: (${newText.length} chars) '$newText'" }

            // Split new text into words. If there are no words, that's an error, otherwise return the words
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

    /**
     * Run the previously created [interpreter] on the provided input, which will return with appended generated text
     * Note that this method may take quite some time to finish, so call this from a background thread
     */
    @WorkerThread
    private fun runInterpreterOn(input: String): String {
        outputBuffer.clear()

        // Run interpreter, which will generate text into outputBuffer
        interpreter.run(input, outputBuffer)

        // Set output buffer limit to current position & position to 0
        outputBuffer.flip()

        // Get bytes from output buffer
        val bytes = ByteArray(outputBuffer.remaining())
        outputBuffer.get(bytes)

        outputBuffer.clear()

        // Return bytes converted to String
        return String(bytes, Charsets.UTF_8)
    }

    /**
     * Load TF Lite model file into memory.
     * The model file is expected in the `src/main/assets` folder, with name configured in [TFLITE_MODEL]
     */
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
        // File name of TF Lite model as expected in the assets folder
        private const val TFLITE_MODEL = "autocomplete.tflite"

        // Size of output buffer for the model to generate text into
        private const val OUTPUT_BUFFER_SIZE = 800

        // Maximum number of words that can be fed into the model
        private const val MAX_INPUT_WORD_COUNT = 1024

        // Maximum number of attempts to generate text that does not contain bad language
        private const val RETRY_COUNT_ON_BAD_LANGUAGE = 3
    }
}