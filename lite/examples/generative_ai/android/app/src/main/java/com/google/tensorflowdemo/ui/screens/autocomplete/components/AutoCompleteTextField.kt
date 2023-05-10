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
 
package com.google.tensorflowdemo.ui.screens.autocomplete.components

import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.input.TextFieldValue
import com.google.tensorflowdemo.R
import com.google.tensorflowdemo.ui.screens.autocomplete.Suggestion
import com.google.tensorflowdemo.ui.theme.InactiveOutlinedTextFieldBorder
import com.google.tensorflowdemo.ui.theme.DarkBlue
import com.google.tensorflowdemo.ui.theme.ActiveOutlinedTextFieldBackground
import com.google.tensorflowdemo.ui.theme.InactiveOutlinedTextFieldBackground
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AutoCompleteTextField(
    inputValue: TextFieldValue,
    inputEnabled: Boolean,
    previousSuggestions: List<Suggestion>,
    onInputValueChange: (TextFieldValue) -> Unit,
    onSuggestionsRemoved: (List<Int>) -> Unit,
    modifier: Modifier = Modifier
) {
    val focusRequester = remember { FocusRequester() }
    val scope = rememberCoroutineScope()
    val colorScheme = MaterialTheme.colorScheme

    SideEffect {
        if (inputEnabled) {
            scope.launch {
                focusRequester.requestFocus()
            }
        }
    }

    OutlinedTextField(
        value = inputValue,
        onValueChange = onInputValueChange,
        enabled = inputEnabled,
        textStyle = MaterialTheme.typography.bodySmall,
        shape = MaterialTheme.shapes.medium,
        placeholder = {
            Text(
                text = stringResource(R.string.input_hint),
                color = MaterialTheme.colorScheme.tertiary,
                modifier = Modifier.alpha(.7f)
            )
        },
        colors = TextFieldDefaults.outlinedTextFieldColors(
            disabledTextColor = MaterialTheme.colorScheme.onSurface,
            unfocusedBorderColor = MaterialTheme.colorScheme.tertiary.copy(alpha = .7f),
            disabledBorderColor = InactiveOutlinedTextFieldBorder,
            focusedBorderColor = DarkBlue,
            containerColor = when {
                inputValue.text.isEmpty() -> InactiveOutlinedTextFieldBackground
                inputEnabled -> ActiveOutlinedTextFieldBackground
                else -> InactiveOutlinedTextFieldBackground
            }
        ),
        keyboardOptions = KeyboardOptions(
            capitalization = KeyboardCapitalization.Sentences,
        ),
        modifier = modifier
            .focusRequester(focusRequester)
    )
}

private fun annotatePreviousSuggestions(
    text: AnnotatedString,
    suggestions: List<Suggestion>,
    colorScheme: ColorScheme,
    onSuggestionsRemoved: (List<Int>) -> Unit
): AnnotatedString {
    val removedSuggestionIds = mutableListOf<Int>()

    val string = buildAnnotatedString {
        append(text)

        for (suggestion in suggestions) {
            val index = text.indexOf(suggestion.text)
            if (index == -1) {
                removedSuggestionIds += suggestion.id
            } else {
                addStyle(
                    style = SpanStyle(color = colorScheme.primary),
                    start = index,
                    end = index + suggestion.text.length
                )
            }
        }
    }

    if (removedSuggestionIds.isNotEmpty()) {
        onSuggestionsRemoved(removedSuggestionIds)
    }

    return string
}
