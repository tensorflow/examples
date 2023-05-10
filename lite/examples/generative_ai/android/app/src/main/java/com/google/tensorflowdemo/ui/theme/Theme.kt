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
 
package com.google.tensorflowdemo.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val TfDemoColorScheme = lightColorScheme(
    primary = Orange,
    secondary = DarkBlue,
    tertiary = DarkGrey,
    surface = VeryLightGrey,
    surfaceVariant = LightGrey,
    outline = MediumGrey,
    error = DarkRed
)

@Composable
fun TensorFlowDemoTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = TfDemoColorScheme,
        typography = TfDemoTypography,
        shapes = TfDemoShapes,
        content = content
    )
}
