package org.tensorflow.lite.examples.textclassification;


import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.*;

import android.content.Context;
import android.support.test.InstrumentationRegistry;
import android.support.test.runner.AndroidJUnit4;

/** Tests of {@link TextClassificationClient} */
@RunWith(AndroidJUnit4.class)
public final class UnitTest {
    private TextClassificationClient client;

    @Before
    public void setUp() {
        // Context of the app under test.
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();

        client = new TextClassificationClient(appContext);
        client.load();
    }

    @Test
    public void loadModelTest() {
        assertNotNull(client.classifier);
    }

    @Test
    public void predictTest() {
        Result positiveText =
                     client
                        .classify("This is an interesting film. My family and I all liked it very much.")
                        .get(0);
        assertEquals("Positive", positiveText.getTitle());
        assertTrue(positiveText.getConfidence() > 0.55);
        Result negativeText =
                client.classify("This film cannot be worse. It is way too boring.").get(0);
        assertEquals("Negative", negativeText.getTitle());
        assertTrue(negativeText.getConfidence() > 0.6);
    }
}