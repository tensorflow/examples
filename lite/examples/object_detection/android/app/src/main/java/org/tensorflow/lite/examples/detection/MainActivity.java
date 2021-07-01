package org.tensorflow.lite.examples.detection;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.googlecode.tesseract.android.TessBaseAPI;

import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    Button btn_camera, btn_folder;
    ImageView imageView;
    TextView txtOCR;
    private String filename = "karnatka.jpeg";
    TessBaseAPI tessBaseAPI;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        btn_camera = findViewById(R.id.Btn_camera);
        btn_folder = findViewById(R.id.Btn_folder);
        imageView = findViewById(R.id.imageDisplay);
        txtOCR = findViewById(R.id.TxtOCr);
        Bitmap bitmap = loadBitmapFromAssets(getBaseContext(), filename);
        imageView.setImageBitmap(bitmap);
        String str = getOcrText(bitmap);
        txtOCR.setText(str);
        btn_camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, DetectorActivity.class);
                startActivity(intent);
            }
        });

        btn_folder.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, Photo_folder_activity.class);
                startActivity(intent);
            }
        });
    }

    protected static Bitmap loadBitmapFromAssets(Context context, String name){
        try{
            InputStream in = context.getAssets().open(name);
            Bitmap bitmap = BitmapFactory.decodeStream(in);
            bitmap = Bitmap.createScaledBitmap(bitmap, 220, 55, true);
            return bitmap;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    private String getOcrText(Bitmap bitmap){
        try{
            tessBaseAPI = new TessBaseAPI();
        } catch (Exception e) {
            e.printStackTrace();
        }
        String dataPath = getExternalFilesDir("/").getPath() + "/";
        tessBaseAPI.setDebug(true);

        tessBaseAPI.init(dataPath, "eng");
        tessBaseAPI.setVariable(TessBaseAPI.VAR_CHAR_WHITELIST, "1234567890abcdefg"+"hijklmnopqrstuvwxyz"+"ABCDEFGHIJKLMNO"+"PQRSTUVWXYZ");
        tessBaseAPI.setVariable(TessBaseAPI.VAR_CHAR_BLACKLIST, "!@#$%^&*()_+=-=[]}{" + ";:'\"\\|~`,./<>?");
        tessBaseAPI.setImage(bitmap);
        String returnString = "no result";
        try {
            returnString = tessBaseAPI.getUTF8Text();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return returnString;
    }
}