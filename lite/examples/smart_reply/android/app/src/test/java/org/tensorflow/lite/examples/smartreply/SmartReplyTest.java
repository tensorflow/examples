/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package org.tensorflow.lite.examples.smartreply;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import androidx.test.core.app.ApplicationProvider;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

@RunWith(RobolectricTestRunner.class)
public class SmartReplyTest {
  private SmartReplyClient client;

  @Before
  public void setUp() throws Exception {
    client = new SmartReplyClient(ApplicationProvider.getApplicationContext());
  }

  @Test
  public void testModelLoaded() {
    client.loadModel();
    assertTrue(client.isLoaded());
  }

  @Test
  public void testUnLoaded() {
    client.loadModel();
    assertTrue(client.isLoaded());
    client.unloadModel();
    assertFalse(client.isLoaded());
  }

  @Test
  public void testPredict() {
    client.loadModel();

    SmartReply[] replies = client.predict(new String[] {"hello"});
    assertNotNull(replies);
    assertTrue("Should have more than 1 messages.", replies.length >= 1);

    final String expectedSubstring = "How are you";
    boolean hasReasonableReply = false;
    for (SmartReply reply : replies) {
      if (reply.getText().contains(expectedSubstring)) {
        hasReasonableReply = true;
      }
    }
    assertTrue("At least one answer should contains `How are you'", hasReasonableReply);
  }

}
