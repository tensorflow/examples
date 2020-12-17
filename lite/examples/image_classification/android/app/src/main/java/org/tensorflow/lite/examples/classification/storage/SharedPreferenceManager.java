package org.tensorflow.lite.examples.classification.storage;

import android.content.Context;
import android.content.SharedPreferences;

import com.google.gson.Gson;

import java.util.ArrayList;

public class SharedPreferenceManager {
    private static SharedPreferences instance;
    private static String SHARED_PREF_NAME = "org.tensorflow.lite.examples.classification.shared_pref";

    private static SharedPreferences getInstance(Context context) {
        if (instance == null) {
            instance = context.getSharedPreferences(SHARED_PREF_NAME, 0);
        }
        return instance;
    }

    public static ItemDetails getItem(Context context, String id) {
        String itemJson = getInstance(context).getString(id, "");
        if (itemJson == null || itemJson.isEmpty()) {
            return null;
        } else {
            return new Gson().fromJson(itemJson, ItemDetails.class);
        }
    }

    public static void addItem(Context context, ItemDetails itemDetails) {
        getInstance(context).edit().putString(itemDetails.getId(), new Gson().toJson(itemDetails)).apply();
    }
}
