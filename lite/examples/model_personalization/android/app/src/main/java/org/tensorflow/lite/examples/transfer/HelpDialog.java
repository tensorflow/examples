/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.examples.transfer;

import android.app.Dialog;
import android.os.Bundle;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatDialogFragment;

/** The help dialog containing a set of instructions on how to run the app. */
public class HelpDialog extends AppCompatDialogFragment {
  @NonNull
  @Override
  public Dialog onCreateDialog(Bundle savedInstanceState) {
    AlertDialog.Builder builder = new AlertDialog.Builder(requireActivity());
    builder
        .setTitle("How to use the app")
        .setMessage(
            "1. There are 4 classes in the app represented by the 4 shapes.\n\n"
                + "2. Each shape can be associated with a category of an object (e.g. car, fruit, "
                + "etc.).\n\n"
                + "3. Take at least 1 image to enable training. For the intended usage, please "
                + "take at least 10 images per category before training.\n\n"
                + "4. Use the 'shape' button while pointing the camera at an object to collect the "
                + "image.\n\n"
                + "If you click the shape button once, it will collect a sample of the object and "
                + "save it. If you long-press the shape button, it will continuously collect "
                + "samples until you release it.\n\n"
                + "5. Press the train button to start training. Once the loss becomes small enough "
                + "(e.g. < 1.0), press the pause button.\n\n"
                + "If the loss doesn't converge to a small number, then press the pause button, "
                + "add more images for each category, and press the resume button.\n\n"
                + "6. Switch to the inference mode and point the camera to some object. Each class "
                + "button will display the probability of which the object falls into that "
                + "category and highlight the most probable one.\n\n"
                + "If training has been done well, the correct category will be "
                + "highlighted. If the object doesn't fall into any of the four categories "
                + "trained, then the app will still try to highlight any one of the closest "
                + "categories with its best guess.\n\n"
                + "7. You can always switch back to the training mode and capture more images.")
        .setPositiveButton("Ok", (dialogInterface, i) -> {});
    return builder.create();
  }
}
