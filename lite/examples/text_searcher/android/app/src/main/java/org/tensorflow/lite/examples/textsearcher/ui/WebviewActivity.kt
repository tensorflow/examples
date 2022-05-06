package org.tensorflow.lite.examples.textsearcher.ui

import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.examples.textsearcher.R

class WebViewActivity : AppCompatActivity() {
    private lateinit var webview: WebView
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_webview)

        webview = findViewById(R.id.webview)
        webview.webViewClient = WebViewClient()
        intent.getStringExtra(getString(R.string.tfe_target_url))?.let { url ->
            webview.loadUrl(url)
        }
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.action_bar_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean = when (item.itemId) {
        R.id.action_close -> {
            // The user chose the "Close" action, so close this activity.
            finish()
            true
        } else -> {
            // If we got here, the user's action was not recognized.
            // Invoke the superclass to handle it.
            super.onOptionsItemSelected(item)
        }
    }
}