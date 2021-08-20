package org.tensorflow.lite.examples.soundclassifier.compose.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material.MaterialTheme
import androidx.compose.material.darkColors
import androidx.compose.material.lightColors
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColorPalette = darkColors(
  primary = gray900,
//  primaryVariant = Purple700,
  secondary = orangeLight,
  onPrimary = Color.White,
//  onSurface = Color(0xFF425066), //gray800,
)

private val LightColorPalette = lightColors(
  primary = Color.White,
//  primaryVariant = Purple700,
  secondary = orangeDeep,
  onPrimary = Color(0xFF425066), //gray900,
//  onSurface = gray900,
)

@Composable
fun JetSoundClassifierTheme(
  darkTheme: Boolean = isSystemInDarkTheme(),
  content: @Composable() () -> Unit
) {
  val colors = if (darkTheme) {
    DarkColorPalette
  } else {
    LightColorPalette
  }

  MaterialTheme(
    colors = colors,
    typography = Typography,
    shapes = Shapes,
    content = content
  )
}