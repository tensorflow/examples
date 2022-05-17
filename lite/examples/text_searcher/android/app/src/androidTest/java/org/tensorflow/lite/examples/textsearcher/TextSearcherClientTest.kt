/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

package org.tensorflow.lite.examples.textsearcher

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.examples.textsearcher.tflite.TextSearcherClient

@RunWith(AndroidJUnit4::class)
class TextSearcherClientTest {
  private lateinit var textSearcherClient: TextSearcherClient

  @Before
  fun setUp() {
    textSearcherClient =
      TextSearcherClient.create(InstrumentationRegistry.getInstrumentation().targetContext)
  }

  @Test
  fun testSearchResult() {
    val results =
      textSearcherClient.search(
        "Shuttle will deliver final U.S. portion of the international space station"
      )

    assert(results.isNotEmpty())
    assertEquals(
      "http://edition.cnn.com:80/2010/US/11/02/space.station.anniversary/index.html",
      results[0].url
    )
    assertEquals(
      "http://www.dailymail.co.uk/news/article-2155446/The-Space-Shuttle-lands-Manhattan-Enterprise-arrives-Intrepid-begin-new-life-New-York-tourist-attraction.html",
      results[1].url
    )
    assertEquals(
      "http://www.dailymail.co.uk/sciencetech/article-2276943/NASA-prepare-Landsat-satellite-launch-reveal-hidden-beauty-Earths-landscapes-space.html",
      results[2].url
    )
    assertEquals(
      "http://edition.cnn.com/2010/TECH/space/01/28/space.shuttle.endeavour/index.html",
      results[3].url
    )
    assertEquals(
      "http://www.dailymail.co.uk/sciencetech/article-2642917/Virgin-FAA-sign-agreement-Spaceport-flights.html",
      results[4].url
    )
  }
}
