package com.google.tensorflowdemo.ui.screens.autocomplete

import android.content.res.Configuration
import androidx.annotation.StringRes
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.ClipboardManager
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.repeatOnLifecycle
import com.google.tensorflowdemo.R
import com.google.tensorflowdemo.data.autocomplete.AutoCompleteService.AutoCompleteInputConfiguration
import com.google.tensorflowdemo.data.autocomplete.AutoCompleteService.AutoCompleteServiceError
import com.google.tensorflowdemo.ui.screens.autocomplete.components.AutoCompleteInfo
import com.google.tensorflowdemo.ui.screens.autocomplete.components.AutoCompleteTextField
import com.google.tensorflowdemo.ui.screens.autocomplete.components.TextControlBar
import com.google.tensorflowdemo.ui.screens.autocomplete.components.WindowSizeSelection
import com.google.tensorflowdemo.ui.theme.TensorFlowDemoTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import org.koin.androidx.compose.getViewModel

@OptIn(ExperimentalComposeUiApi::class)
@Composable
fun AutoCompleteScreen(
    onShowToast: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    val viewmodel = getViewModel<AutoCompleteViewModel>()

    val textValue = rememberSaveable(stateSaver = TextFieldValue.Saver) { mutableStateOf(TextFieldValue(annotatedString = AnnotatedString(""))) }
    val barState by viewmodel.textBarState.collectAsState()
    val inputFieldEnabled by viewmodel.inputFieldEnabled.collectAsStateWithLifecycle()
    val windowSizeConfiguration by remember { mutableStateOf(viewmodel.windowSizeConfiguration) }
    val clipboardManager: ClipboardManager = LocalClipboardManager.current
    val keyboardController = LocalSoftwareKeyboardController.current
    val uriHandler = LocalUriHandler.current

    fun showToast(@StringRes id: Int) {
        keyboardController?.hide()

        onShowToast(id)
    }

    AutoCompleteScreenContent(
        inputValue = textValue.value,
        inputEnabled = inputFieldEnabled,
        onInputValueChange = { value ->
            textValue.value = value
            viewmodel.isTextEmpty = value.text.isEmpty()
        },
        barState = barState,
        inputConfiguration = windowSizeConfiguration,
        previousSuggestions = viewmodel.previousSuggestions,
        onClear = {
            textValue.value = TextFieldValue(AnnotatedString(""))
            viewmodel.onClearInput()
        },
        onCopy = {
            clipboardManager.setText(textValue.value.annotatedString)

            onShowToast(R.string.text_copied)
        },
        onGenerate = { viewmodel.onGenerateAutoComplete(textValue.value.text) },
        onRetry = viewmodel::onRetryGenerateAutoComplete,
        onAccept = {
            viewmodel.onAcceptSuggestion()
            textValue.value = TextFieldValue(
                text = textValue.value.text,
                selection = TextRange(textValue.value.text.length)
            )
        },
        onWindowSizeChange = viewmodel::onWindowSizeChange,
        onSuggestionsRemoved = viewmodel::removeMissingSuggestions,
        onLinkoutSelect = { uriHandler.openUri("https://github.com/keras-team/keras-nlp") },
        modifier = modifier
    )

    val lifecycle = LocalLifecycleOwner.current.lifecycle
    val colorScheme = MaterialTheme.colorScheme

    LaunchedEffect(key1 = Unit) {
        lifecycle.repeatOnLifecycle(state = Lifecycle.State.STARTED) {
            launch {
                viewmodel.suggestion.collectLatest { words ->
                    words?.let {
                        animateSuggestion(textValue, words, colorScheme) {
                            viewmodel.onSuggestionReceived()
                        }
                    }
                }
            }
            launch {
                viewmodel.resetInputText.collectLatest { resetText ->
                    resetText?.let { text ->
                        textValue.value = TextFieldValue(
                            annotatedString = AnnotatedString(text),
                            selection = TextRange(text.length)
                        )

                        viewmodel.onResetReceived()
                    }
                }
            }
            launch {
                viewmodel.error.collectLatest { error ->
                    showToast(getErrorMessage(error))
                }
            }
        }
    }
}

suspend fun animateSuggestion(
    textValueState: MutableState<TextFieldValue>,
    words: List<String>,
    colorScheme: ColorScheme,
    onAnimationComplete: () -> Unit
) {
    val builder = AnnotatedString.Builder(textValueState.value.annotatedString)

    val stylePos = builder.pushStyle(
        SpanStyle(
            color = colorScheme.primary,
            fontWeight = FontWeight.Bold
        )
    )

    for (word in words) {
        builder.append(word)

        val annotatedString = builder.toAnnotatedString()
        textValueState.value = TextFieldValue(
            annotatedString = annotatedString,
            selection = TextRange(annotatedString.length)
        )
        delay(100)
    }

    builder.pop(stylePos)

    onAnimationComplete()
}

@Composable
fun AutoCompleteScreenContent(
    inputValue: TextFieldValue,
    inputEnabled: Boolean,
    onInputValueChange: (TextFieldValue) -> Unit,
    barState: TextEditBarState,
    previousSuggestions: List<Suggestion>,
    inputConfiguration: AutoCompleteInputConfiguration,
    onClear: () -> Unit,
    onCopy: () -> Unit,
    onGenerate: () -> Unit,
    onRetry: () -> Unit,
    onAccept: () -> Unit,
    onWindowSizeChange: (Int) -> Unit,
    onSuggestionsRemoved: (List<Int>) -> Unit,
    onLinkoutSelect: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .background(Color.White)
            .fillMaxHeight()
    ) {
        Column(modifier = Modifier.fillMaxHeight()) {
            Column(
                modifier = modifier
                    .background(Color.White)
                    .padding(start = 16.dp, end = 16.dp, top = 20.dp)
            ) {
                AutoCompleteTextField(
                    inputValue = inputValue,
                    inputEnabled = inputEnabled,
                    previousSuggestions = previousSuggestions,
                    onInputValueChange = onInputValueChange,
                    onSuggestionsRemoved = onSuggestionsRemoved,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(250.dp)
                        .padding(bottom = 16.dp),
                )
                TextControlBar(
                    state = barState,
                    onClearClick = onClear,
                    onGenerateClick = onGenerate,
                    onCopyClick = onCopy,
                    onAccept = onAccept,
                    onRetry = onRetry
                )
            }
            Spacer(modifier = Modifier.weight(1f))
            WindowSizeSelection(
                inputConfiguration = inputConfiguration,
                onWindowValueChange = onWindowSizeChange,
                modifier = Modifier.padding(bottom = 32.dp, start = 16.dp, end = 16.dp)
            )
            AutoCompleteInfo(
                onLinkoutSelect = onLinkoutSelect,
                modifier = Modifier
                    .padding(horizontal = 8.dp, vertical = 8.dp)
            )
        }
    }
}

private fun getErrorMessage(error: AutoCompleteServiceError) = when (error) {
    AutoCompleteServiceError.MODEL_NOT_INITIALIZED -> R.string.error_model_not_initialized
    AutoCompleteServiceError.NO_SUGGESTIONS -> R.string.error_no_suggestion_found
    AutoCompleteServiceError.MODEL_FILE_NOT_FOUND -> R.string.error_model_not_found
    AutoCompleteServiceError.BAD_LANGUAGE -> R.string.error_input_contains_bad_language
}

@Preview(uiMode = Configuration.UI_MODE_NIGHT_NO, showBackground = true)
@Composable
fun PreviewAutoCompleteScreen() {
    TensorFlowDemoTheme {
        val inputValue by remember { mutableStateOf(TextFieldValue()) }
        AutoCompleteScreenContent(
            inputValue = inputValue,
            onInputValueChange = {},
            inputEnabled = true,
            inputConfiguration = AutoCompleteInputConfiguration(5, 50, 20),
            previousSuggestions = listOf(),
            onClear = {},
            onCopy = {},
            onGenerate = {},
            onRetry = {},
            onAccept = {},
            onWindowSizeChange = {},
            onSuggestionsRemoved = {},
            onLinkoutSelect = {},
            barState = initialControlBarState,
        )
    }
}