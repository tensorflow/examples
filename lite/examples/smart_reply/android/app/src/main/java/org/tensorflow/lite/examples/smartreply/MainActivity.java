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

import android.os.Bundle;
import android.os.Handler;
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.TextView;

/**
 * The main (and only) activity of this demo app. Displays a text box which updates as messages are
 * received.
 */
public class MainActivity extends AppCompatActivity {
  private static final String TAG = "SmartReplyDemo";
  private SmartReplyClient client;

  private TextView messageTextView;
  private EditText messageInput;
  private ScrollView scrollView;

  private Handler handler;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    Log.v(TAG, "onCreate");
    setContentView(R.layout.tfe_sr_main_activity);

    client = new SmartReplyClient(getApplicationContext());
    handler = new Handler();

    scrollView = findViewById(R.id.scroll_view);
    messageTextView = findViewById(R.id.message_text);

    messageInput = findViewById(R.id.message_input);
    messageInput.setOnKeyListener(
        (view, keyCode, keyEvent) -> {
          if (keyCode == KeyEvent.KEYCODE_ENTER && keyEvent.getAction() == KeyEvent.ACTION_UP) {
            // Send message when pressing Enter on keyboard.
            send(messageInput.getText().toString());
            return true;
          }
          return false;
        });

    Button sendButton = findViewById(R.id.send_button);
    sendButton.setOnClickListener((View v) -> send(messageInput.getText().toString()));
  }

  @Override
  protected void onStart() {
    super.onStart();
    Log.v(TAG, "onStart");
    handler.post(
        () -> {
          client.loadModel();
        });
  }

  @Override
  protected void onStop() {
    super.onStop();
    Log.v(TAG, "onStop");
    handler.post(
        () -> {
          client.unloadModel();
        });
  }

  private void send(final String message) {
    handler.post(
        () -> {
          StringBuilder textToShow = new StringBuilder();
          textToShow.append("Input: ").append(message).append("\n\n");

          // Get suggested replies from the model.
          SmartReply[] ans = client.predict(new String[] {message});
          for (SmartReply reply : ans) {
            textToShow.append("Reply: ").append(reply.getText()).append("\n");
          }
          textToShow.append("------").append("\n");

          runOnUiThread(
              () -> {
                // Show the message and suggested replies on screen.
                messageTextView.append(textToShow);

                // Clear the input box
                messageInput.setText(null);

                // Scroll to the bottom to show latest entry's classification result.
                scrollView.post(() -> scrollView.fullScroll(View.FOCUS_DOWN));
              });
        });
  }
}
