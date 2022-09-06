/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imageclassification;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.vision.classifier.Classifications;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class ImageClassificationTest {
    List<Category> controlCategories = new ArrayList<>(Arrays.asList(
            new Category("cup", 0.7578125f))
    );

    @Test
    public void classificationResultsShouldNotChange() {
        ImageClassifierHelper helper = ImageClassifierHelper.create(
                InstrumentationRegistry.getInstrumentation().getContext(),
                new ImageClassifierHelper.ClassifierListener() {
                    @Override
                    public void onError(String error) {
                        // no-op
                    }

                    @Override
                    public void onResults(
                            List<Classifications> results,
                            long inferenceTime
                    ) {
                        assertNotNull(results.get(0));

                        // Verify that the classified data and control
                        // data have the same number of categories
                        assertEquals(controlCategories.size(),
                                results.get(0).getCategories().size());

                        // Loop through the categories
                        for (int i = 0; i < results.size(); i++) {
                            // Verify that the labels are consistent
                            assertEquals(
                                    controlCategories.get(i).getLabel(),
                                    results.get(0).getCategories().get(i).getLabel()
                            );
                        }
                    }
                });
        helper.setThreshold(0.0f);
        helper.classify(loadImage("coffee.jpg"), 0);
    }

    private Bitmap loadImage(String fileName) {
        AssetManager assetManager = InstrumentationRegistry
                .getInstrumentation()
                .getContext()
                .getAssets();
        try {
            InputStream inputStream = assetManager.open(fileName);
            return BitmapFactory.decodeStream(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}