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

import android.content.res.Configuration
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import com.google.tensorflowdemo.R
import com.google.tensorflowdemo.ui.theme.TensorFlowDemoTheme

@Composable
fun AutoCompleteInfo(
    onLinkoutSelect: () -> Unit,
    modifier: Modifier = Modifier
) {
    TextButton(
        onClick = onLinkoutSelect,
        modifier = modifier.fillMaxWidth()
    ) {
        Spacer(modifier = Modifier.weight(1f))
        Text(
            text = stringResource(R.string.about_title),
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.tertiary,
        )
        Spacer(modifier = Modifier.weight(1f))
    }
}

@Preview(uiMode = Configuration.UI_MODE_NIGHT_NO, showBackground = true)
@Composable
fun PreviewAutCompleteInfo() {
    TensorFlowDemoTheme {
        AutoCompleteInfo({})
    }
}