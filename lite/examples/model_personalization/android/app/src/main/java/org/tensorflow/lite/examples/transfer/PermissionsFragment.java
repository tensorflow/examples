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

package org.tensorflow.lite.examples.transfer;

import android.Manifest.permission;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.widget.Toast;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

/**
 * The sole purpose of this fragment is to request the necessary permissions.
 * It does not create a view.
 */
public class PermissionsFragment extends Fragment {

  private static final int PERMISSIONS_REQUEST_CODE = 10;
  private static final String[] PERMISSIONS_REQUIRED = { permission.CAMERA };

  private PermissionsAcquiredListener callback;

  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    if (!hasPermissions()) {
      requestPermissions(PERMISSIONS_REQUIRED, PERMISSIONS_REQUEST_CODE);
    } else {
      callback.onPermissionsAcquired();
    }
  }

  private boolean hasPermissions() {
    for (String permission : PERMISSIONS_REQUIRED) {
      if (ContextCompat.checkSelfPermission(requireContext(), permission)
          != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }

    return true;
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == PERMISSIONS_REQUEST_CODE) {
      if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        Toast.makeText(getContext(), "Camera permission granted", Toast.LENGTH_LONG).show();
        callback.onPermissionsAcquired();
      } else {
        Toast.makeText(getContext(), "Camera permission denied", Toast.LENGTH_LONG).show();
      }
    }
  }

  public void setOnPermissionsAcquiredListener(PermissionsAcquiredListener callback) {
    this.callback = callback;
  }

  /**
   * Should be implemented by the host activity to get notified about permissions being acquired.
   */
  public interface PermissionsAcquiredListener {
    void onPermissionsAcquired();
  }
}
