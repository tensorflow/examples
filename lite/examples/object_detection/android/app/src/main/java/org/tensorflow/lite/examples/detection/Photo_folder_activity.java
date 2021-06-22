package org.tensorflow.lite.examples.detection;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;

import org.tensorflow.lite.examples.detection.utils.MarginDecoration;
import org.tensorflow.lite.examples.detection.utils.PicHolder;
import org.tensorflow.lite.examples.detection.utils.itemClickListener;
import org.tensorflow.lite.examples.detection.utils.imageFolder;
import org.tensorflow.lite.examples.detection.utils.pictureFacer;
import org.tensorflow.lite.examples.detection.utils.pictureFolderAdapter;

import java.util.ArrayList;

public class Photo_folder_activity extends AppCompatActivity implements itemClickListener {

    RecyclerView folderRecycler;
    TextView empty;
    private static final int MY_PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE = 1;

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_photo_folder);
        if(ContextCompat.checkSelfPermission(Photo_folder_activity.this,
                Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED)
            ActivityCompat.requestPermissions(Photo_folder_activity.this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    MY_PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE);

        empty = findViewById(R.id.empty);
        folderRecycler = findViewById(R.id.folderRecycler);
        folderRecycler.addItemDecoration(new MarginDecoration(this));
        folderRecycler.hasFixedSize();
        ArrayList<imageFolder> foldeersArray = getPicturePaths();

        if(foldeersArray.isEmpty()){
            empty.setVisibility(View.VISIBLE);
        }else{
            RecyclerView.Adapter folderAdapter = new pictureFolderAdapter(foldeersArray, Photo_folder_activity.this, this);
            folderRecycler.setAdapter(folderAdapter);
        }
        changeStatusBarColor();
    }

    private ArrayList<imageFolder> getPicturePaths(){
        ArrayList<imageFolder> picFolders = new ArrayList<>();
        ArrayList<String> picPaths = new ArrayList<>();
        Uri allImagesUri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;
        String[] projection = {MediaStore.Images.ImageColumns.DATA, MediaStore.Images.Media.DISPLAY_NAME,
                MediaStore.Images.Media.BUCKET_DISPLAY_NAME, MediaStore.Images.Media.BUCKET_ID};
        Log.i("projection", projection.toString());
        Cursor cursor = this.getContentResolver().query(allImagesUri, projection, null, null, null);
        try{
            if(cursor != null){
                cursor.moveToFirst();
            }
            do{
                imageFolder img_folds = new imageFolder();
                String name = cursor.getString(cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DISPLAY_NAME));
                String folder = cursor.getString(cursor.getColumnIndexOrThrow(MediaStore.Images.Media.BUCKET_DISPLAY_NAME));
                String datapath = cursor.getString(cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA));
                String folderpaths = datapath.substring(0, datapath.lastIndexOf(folder + "/"));
                folderpaths = folderpaths + folder + "/";

                if(!picPaths.contains(folderpaths)){
                    picPaths.add(folderpaths);
                    img_folds.setPath(folderpaths);
                    img_folds.setFolderName(folder);
                    img_folds.setFirstPic(datapath);
                    img_folds.addpics();
                    picFolders.add(img_folds);
                }else{
                    for(int i = 0; i<picFolders.size(); i++){
                        if(picFolders.get(i).getPath().equals(folderpaths)){
                            picFolders.get(i).setFirstPic(datapath);
                            picFolders.get(i).addpics();
                        }
                    }
                }
            }while (cursor.moveToNext());
        }catch (Exception e){
            e.printStackTrace();
        }

        for(int i = 0; i<picFolders.size(); i++){
            Log.d("picture folders", picFolders.get(i).getFolderName()+" and path = "+picFolders.get(i).getPath()+" "+picFolders.get(i).getNumberOfPics());
        }
        return picFolders;
    }

    @Override
    public void onPicClicked(PicHolder holder, int position, ArrayList<pictureFacer> pics) {

    }

    @Override
    public void onPicClicked(String pictureFolderPath, String folderName) {
        Intent move = new Intent(Photo_folder_activity.this, ImageDisplay.class);
        move.putExtra("folderPath", pictureFolderPath);
        move.putExtra("folderName", folderName);
        Log.d("clicked folder:" , "\nfolder path : " + pictureFolderPath + "\nfolder name : " + folderName);
        startActivity(move);
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private void changeStatusBarColor(){
        Window window = this.getWindow();
        window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
        window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
        window.setStatusBarColor(ContextCompat.getColor(getApplicationContext(), R.color.black));
    }
}