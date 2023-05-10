package com.google.tensorflowdemo.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.FixedScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.google.tensorflowdemo.R
import com.google.tensorflowdemo.ui.theme.TensorFlowDemoTheme

@Composable
fun HeaderBar(
    label: String,
    textOffset: Dp,
    modifier:Modifier = Modifier
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .background(Color.White)
    ) {
        Image(
            painter = painterResource(id = R.drawable.header_background),
            contentDescription = stringResource(R.string.header_background_desc),
            contentScale = FixedScale(.47f),
        )
        Text(
            text = label,
            color = MaterialTheme.colorScheme.secondary,
            style = MaterialTheme.typography.displayLarge,
            modifier = Modifier
                .align(Alignment.Center)
                .offset(y = textOffset)
        )
        Divider(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .align(Alignment.BottomStart)
        )

    }
}

@Preview
@Composable
fun PreviewHeaderBar() {
    TensorFlowDemoTheme {
        HeaderBar(
            label = "Autocomplete",
            textOffset = 20.dp
        )
    }
}