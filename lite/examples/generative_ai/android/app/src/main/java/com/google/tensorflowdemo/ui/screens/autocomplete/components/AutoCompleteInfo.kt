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