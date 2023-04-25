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
