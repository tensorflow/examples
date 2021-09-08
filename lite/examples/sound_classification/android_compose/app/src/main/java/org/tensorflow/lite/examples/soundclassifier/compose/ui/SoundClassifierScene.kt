package org.tensorflow.lite.examples.soundclassifier.compose.ui

import android.content.res.Configuration
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Devices
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import org.tensorflow.lite.examples.soundclassifier.compose.R
import org.tensorflow.lite.examples.soundclassifier.compose.ui.theme.JetSoundClassifierTheme
import org.tensorflow.lite.examples.soundclassifier.compose.ui.theme.gray800
import org.tensorflow.lite.examples.soundclassifier.compose.ui.theme.orange500
import org.tensorflow.lite.examples.soundclassifier.compose.ui.theme.progressColorPairs
import org.tensorflow.lite.support.label.Category

@Composable
fun SoundClassifierScene(viewModel: SoundClassifierViewModel) {
  val classifierEnabled by viewModel.classifierEnabled.collectAsState()
  val classificationInterval by viewModel.classificationInterval.collectAsState()
  val probabilities by viewModel.probabilities.collectAsState()

  JetSoundClassifierTheme {
    Surface(color = MaterialTheme.colors.background) {
      Scaffold(
        topBar = {
          TopAppBar(
            title = {
              Image(
                painter = painterResource(R.drawable.tfl2_logo),
                contentDescription = stringResource(id = R.string.tensorflow_lite_logo),
                modifier = Modifier.sizeIn(maxWidth = 180.dp),
              )
            },
            backgroundColor = MaterialTheme.colors.primarySurface
          )
        }) { innerPadding ->
        SoundClassifierBody(
          probabilities = probabilities,
          classifierEnabled = classifierEnabled,
          interval = classificationInterval,
          onClassifierToggle = viewModel::setClassifierEnabled,
          onIntervalChanged = viewModel::setClassificationInterval,
          modifier = Modifier.padding(innerPadding),
        )
      }
    }
  }

}

@Composable
private fun SoundClassifierBody(
  probabilities: List<Category>,
  classifierEnabled: Boolean,
  interval: Long,
  onClassifierToggle: (Boolean) -> Unit,
  onIntervalChanged: (Long) -> Unit,
  modifier: Modifier = Modifier,
) {
  Column(
    modifier = modifier.padding(16.dp)
  ) {
    ControlPanel(
      inputEnabled = classifierEnabled,
      interval = interval,
      onInputChanged = onClassifierToggle,
      onIntervalChanged = onIntervalChanged,
    )

    Divider(modifier = Modifier.padding(vertical = 24.dp))

    if (classifierEnabled) {
      probabilities.let { itemList ->
        LazyColumn {
          itemsIndexed(
            items = itemList,
            key = { _, item -> item.label }
          ) { index, item ->
            ProbabilityItem(text = item.label, progress = item.score, index = index)
          }
        }
      }
    }
  }
}

@Composable
fun ControlPanel(
  inputEnabled: Boolean,
  interval: Long,
  onInputChanged: ((Boolean) -> Unit) = {},
  onIntervalChanged: (Long) -> Unit = {},
) {
  Row {
    val labelText = stringResource(id = R.string.label_input)
    Text(labelText, style = MaterialTheme.typography.body1)
    Switch(
      checked = inputEnabled,
      onCheckedChange = onInputChanged,
      modifier = Modifier.padding(start = 16.dp),
      colors = SwitchDefaults.colors(checkedThumbColor = orange500)
    )
  }
  Spacer(modifier = Modifier.height(12.dp))
  Row(verticalAlignment = Alignment.CenterVertically) {
    val labelText = stringResource(id = R.string.label_classification_interval)
    Text(labelText, style = MaterialTheme.typography.body1)
    Slider(
      enabled = inputEnabled,
      value = interval / 1000f,
      onValueChange = { onIntervalChanged((it * 1000L).toLong()) },
      modifier = Modifier
        .padding(start = 8.dp)
        .fillMaxWidth(),
      colors = SliderDefaults.colors(
        thumbColor = gray800,
        activeTrackColor = gray800,
      )
    )
  }
}

@Composable
fun ProbabilityItem(text: String, progress: Float, index: Int = 0) {
  val indicatorColor = progressColorPairs[index % 3].second
  val backgroundColor = progressColorPairs[index % 3].first

  Row(
    verticalAlignment = Alignment.CenterVertically
  ) {
    Text(
      text,
      modifier = Modifier.width(92.dp),
      style = MaterialTheme.typography.body2,
    )
    LinearProgressIndicator(
      progress = progress,
      modifier = Modifier
        .height(52.dp)
        .padding(start = 12.dp, top = 16.dp, end = 16.dp, bottom = 16.dp)
        .clip(MaterialTheme.shapes.medium)
        .fillMaxWidth(),
      color = indicatorColor,
      backgroundColor = backgroundColor
    )
  }
}

val DummyCategories = listOf(
  Category("Background Noise", 0.8f),
  Category("Clap", 0.8f),
  Category("Snap", 0.8f),
)

@Preview(name = "Day mode, small device", widthDp = 360, heightDp = 640)
@Preview(
  name = "Night mode, small device", widthDp = 360, heightDp = 640,
  uiMode = Configuration.UI_MODE_NIGHT_YES,
)
@Preview(name = "Day mode", device = Devices.PIXEL_4_XL)
@Composable
fun Preview() {
  JetSoundClassifierTheme {
    Surface(color = MaterialTheme.colors.background) {
      SoundClassifierBody(
        probabilities = DummyCategories,
        classifierEnabled = true,
        interval = 500L,
        onClassifierToggle = {},
        onIntervalChanged = {},
      )
    }
  }
}