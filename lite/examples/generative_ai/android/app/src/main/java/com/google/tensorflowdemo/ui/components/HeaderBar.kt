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