package org.tensorflow.lite.examples.bertqa;

import android.content.Context;

import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.runner.AndroidJUnit4;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.tensorflow.lite.examples.bertqa.tflite.QaClient;
import org.tensorflow.lite.task.text.qa.QaAnswer;

import static org.junit.Assert.*;


/** Tests of {@link QaClient} */
@RunWith(AndroidJUnit4.class)
public final class UnitTest {
    private QaClient client;
    private String contextOfTheQuestion = "Nikola Tesla (Serbian Cyrillic: Никола Тесла; 10 July 1856 – 7 January 1943) was a Serbian American inventor, electrical engineer, mechanical engineer, physicist, and futurist best known for his contributions to the design of the modern alternating current (AC) electricity supply system.";

    @Before
    public void setUp() {
        // Context of the app under test.
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();

        client = new QaClient(appContext);
        client.loadModel();
    }

    @Test
    public void loadModelTest() {
        assertNotNull(client.answerer);
    }

    @Test
    public void predictTest() {

        QaAnswer answer = client.predict("In what year was Nikola Tesla born? ", contextOfTheQuestion).get(0);

        assertEquals("1856", answer.text);
    }
}