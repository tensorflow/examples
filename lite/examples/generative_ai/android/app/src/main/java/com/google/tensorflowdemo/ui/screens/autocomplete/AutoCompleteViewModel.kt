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
 
package com.google.tensorflowdemo.ui.screens.autocomplete

import androidx.compose.runtime.mutableStateListOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.google.tensorflowdemo.data.autocomplete.AutoCompleteService
import com.google.tensorflowdemo.data.autocomplete.AutoCompleteService.AutoCompleteResult
import com.google.tensorflowdemo.data.autocomplete.AutoCompleteService.AutoCompleteServiceError
import com.google.tensorflowdemo.data.autocomplete.AutoCompleteService.InitModelResult
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class AutoCompleteViewModel(
    private val autoCompleteService: AutoCompleteService
) : ViewModel() {

    private var currentSuggestionInputText: String = ""
    private var currentSuggestionText: String? = null
    private var windowSize = 0
    private var initModelError: InitModelResult.Error? = null
    private var previousSuggestionsIndex = 0
    private val isGenerating = MutableStateFlow(false)
    private val hasGenerated = MutableStateFlow(false)
    private val isSuggesting = MutableStateFlow(false)
    private val isModelInitialized = MutableStateFlow(false)

    init {
        if (!autoCompleteService.isInitialized) {
            viewModelScope.launch {
                val result = autoCompleteService.initModel()
                if (result is InitModelResult.Error) {
                    initModelError = result
                }
                isModelInitialized.value = true
            }
        } else {
            isModelInitialized.value = true
        }
    }

    private val _isTextEmpty = MutableStateFlow(true)
    var isTextEmpty: Boolean = true
        set(value) {
            field = value
            _isTextEmpty.value = value

            // Clear previous suggestions if the input text is empty
            if (value) {
                _previousSuggestions.clear()
            }
        }

    val windowSizeConfiguration = autoCompleteService.inputConfiguration

    /**
     * State flow to reset text to previous value.
     * Needs to be acknowledged by calling [onResetReceived] since the previous value can be the same
     */
    private val _resetInputText = MutableStateFlow<String?>(null)
    val resetInputText: StateFlow<String?>
        get() = _resetInputText

    fun onResetReceived() {
        _resetInputText.value = null
    }

    /**
     * State flow containing most recent suggestion from model, as list of words
     * Needs to be acknowledged by calling [onSuggestionReceived]
     */
    private val _suggestion = MutableStateFlow<List<String>?>(null)
    val suggestion: StateFlow<List<String>?>
        get() = _suggestion

    /**
     * Shared flow exposing errors from autocomplete service
     */
    private val _error = MutableSharedFlow<AutoCompleteServiceError>()
    val error: SharedFlow<AutoCompleteServiceError>
        get() = _error

    /**
     * State flow exposing previously made suggestions
     */
    private val _previousSuggestions = mutableStateListOf<Suggestion>()
    val previousSuggestions: List<Suggestion>
        get() = _previousSuggestions

    /**
     * State flow exposing whether Clear CTA should be enabled
     */
    private val clearEnabled = combine(isGenerating, _isTextEmpty) { isGenerating, isEmpty ->
        !isGenerating && !isEmpty
    }

    /**
     * State flow exposing whether Generate CTA should be enabled
     */
    private val generateEnabled = combine(isGenerating, _isTextEmpty) { isGenerating, isEmpty ->
        !isGenerating && !isEmpty
    }

    /**
     * State flow exposing whether Copy CTA should be enabled
     */
    private val copyEnabled = combine(isGenerating, hasGenerated) { isGenerating, hasGenerated ->
        !isGenerating && hasGenerated
    }

    /**
     * State flow exposing edit bar state for Clear, Generate & Copy CTAs & generation process state
     */
    private val editingBarState = combine(clearEnabled, generateEnabled, copyEnabled, isGenerating) { clear, generate, copy, generating ->
        TextEditBarState.Editing(
            clearEnabled = clear,
            generateEnabled = generate,
            copyEnabled = copy,
            generating = generating,
        )
    }

    /**
     * State flow exposing edit bar state & whether a suggestion is active
     */
    val textBarState = combine(editingBarState, isSuggesting) { editState, suggesting ->
        if (suggesting) TextEditBarState.Suggesting
        else editState
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.Lazily,
        initialValue = initialControlBarState
    )

    /**
     * State flow exposing whether input by user should be possible
     */
    val inputFieldEnabled = combine(isModelInitialized, isGenerating, isSuggesting) { initialized, generating, suggesting ->
        initialized && !generating && !suggesting
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.Lazily,
        initialValue = true
    )

    fun onClearInput() {
        isGenerating.value = false
        hasGenerated.value = false
        _isTextEmpty.value = true
        currentSuggestionInputText = ""

        _previousSuggestions.clear()
    }

    fun onWindowSizeChange(size: Int) {
        windowSize = size
    }

    fun onRetryGenerateAutoComplete() {
        _resetInputText.value = currentSuggestionInputText

        isSuggesting.value = false
        currentSuggestionText = null

        onGenerateAutoComplete(currentSuggestionInputText)
    }

    fun onAcceptSuggestion() {
        currentSuggestionText?.let { text ->
            _previousSuggestions += Suggestion(
                text = text,
                id = previousSuggestionsIndex++
            )
        }

        isSuggesting.value = false
        currentSuggestionText = null
    }

    fun removeMissingSuggestions(ids: List<Int>) {
        for (id in ids) {
            _previousSuggestions.removeIf { suggestion -> suggestion.id == id }
        }
    }

    fun onGenerateAutoComplete(text: String) {
        initModelError?.let { error ->
            viewModelScope.launch {
                _error.emit(error.error)
            }
            return
        }

        currentSuggestionInputText = text

        isGenerating.value = true

        viewModelScope.launch {
            when (val result = autoCompleteService.getSuggestion(text, applyWindow = true, windowSize = windowSize)) {
                is AutoCompleteResult.Success -> {
                    _suggestion.value = result.words
                    currentSuggestionText = result.words.joinToString(separator = "")
                }

                is AutoCompleteResult.Error -> {
                    _error.emit(result.error)

                    isGenerating.value = false
                }
            }
        }
    }

    fun onSuggestionReceived() {
        _suggestion.value = null

        isGenerating.value = false
        hasGenerated.value = true
        isSuggesting.value = true
    }
}

sealed class TextEditBarState {
    data class Editing(
        val clearEnabled: Boolean,
        val generateEnabled: Boolean,
        val copyEnabled: Boolean,
        val generating: Boolean,
    ) : TextEditBarState()

    object Suggesting : TextEditBarState()
}

val initialControlBarState: TextEditBarState = TextEditBarState.Editing(
    clearEnabled = false,
    generateEnabled = false,
    copyEnabled = false,
    generating = false
)

data class Suggestion(
    val text: String,
    val id: Int
)