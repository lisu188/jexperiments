package pl.lisu188.speechrecorder;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity {
    public static final String ACTION_RESUME_AFTER_BOOT = "pl.lisu188.speechrecorder.RESUME_AFTER_BOOT";
    private static final int REQUEST_PERMISSIONS = 1001;
    private TextView status;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(buildUi());
        if (ACTION_RESUME_AFTER_BOOT.equals(getIntent().getAction())) {
            requestAndStart();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (status != null) {
            boolean enabled = getSharedPreferences("recorder", MODE_PRIVATE).getBoolean("enabled", false);
            status.setText(enabled ? "Status: nasłuchiwanie aktywne" : "Status: zatrzymany");
        }
    }

    private View buildUi() {
        int pad = dp(24);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);
        root.setGravity(Gravity.CENTER_HORIZONTAL);

        TextView title = new TextView(this);
        title.setText("Speech Recorder");
        title.setTextSize(28);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(title, matchWrap());

        TextView description = new TextView(this);
        description.setText("Nasłuchuje lokalnie i zapisuje tylko fragmenty, w których wykryje mowę. Audio nie opuszcza telefonu.");
        description.setTextSize(16);
        description.setPadding(0, dp(18), 0, dp(18));
        root.addView(description, matchWrap());

        status = new TextView(this);
        status.setText("Status: zatrzymany");
        status.setTextSize(18);
        status.setTypeface(Typeface.DEFAULT_BOLD);
        status.setPadding(0, 0, 0, dp(18));
        root.addView(status, matchWrap());

        Button start = new Button(this);
        start.setText("START");
        start.setOnClickListener(v -> requestAndStart());
        root.addView(start, matchWrap());

        Button stop = new Button(this);
        stop.setText("STOP");
        stop.setOnClickListener(v -> stopRecorder());
        LinearLayout.LayoutParams stopParams = matchWrap();
        stopParams.topMargin = dp(10);
        root.addView(stop, stopParams);

        TextView settings = new TextView(this);
        settings.setText("Ustawienia: 5 s bufora przed mową, 8 s ciszy kończącej klip, WAV 16 kHz mono.\nNagrania: Music/SpeechRecorder\nTryb trwały: foreground service + START_STICKY.");
        settings.setTextSize(14);
        settings.setPadding(0, dp(24), 0, dp(16));
        root.addView(settings, matchWrap());

        TextView privacy = new TextView(this);
        privacy.setText("Nagrywanie mikrofonu jest sygnalizowane przez Androida. Używaj zgodnie z prawem i zasadami prywatności osób znajdujących się w otoczeniu.");
        privacy.setTextSize(13);
        root.addView(privacy, matchWrap());

        Button appSettings = new Button(this);
        appSettings.setText("USTAWIENIA APLIKACJI");
        appSettings.setOnClickListener(v -> {
            Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
            intent.setData(android.net.Uri.parse("package:" + getPackageName()));
            startActivity(intent);
        });
        LinearLayout.LayoutParams appSettingsParams = matchWrap();
        appSettingsParams.topMargin = dp(20);
        root.addView(appSettings, appSettingsParams);

        Button batterySettings = new Button(this);
        batterySettings.setText("OPTYMALIZACJA BATERII");
        batterySettings.setOnClickListener(v -> startActivity(new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)));
        LinearLayout.LayoutParams batteryParams = matchWrap();
        batteryParams.topMargin = dp(10);
        root.addView(batterySettings, batteryParams);

        TextView persistence = new TextView(this);
        persistence.setText("Po uruchomieniu Android utrzymuje usługę w tle i może odtworzyć ją po ubiciu procesu. Force stop, odebranie dostępu do mikrofonu albo ograniczenia systemowe nadal mogą zatrzymać aplikację.");
        persistence.setTextSize(13);
        persistence.setPadding(0, dp(16), 0, 0);
        root.addView(persistence, matchWrap());

        return root;
    }

    private void requestAndStart() {
        List<String> missing = new ArrayList<>();
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.RECORD_AUDIO);
        }
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.POST_NOTIFICATIONS);
        }
        if (!missing.isEmpty()) {
            requestPermissions(missing.toArray(new String[0]), REQUEST_PERMISSIONS);
            return;
        }
        startRecorder();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != REQUEST_PERMISSIONS) {
            return;
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            startRecorder();
        } else {
            Toast.makeText(this, "Bez dostępu do mikrofonu aplikacja nie może działać.", Toast.LENGTH_LONG).show();
        }
    }

    private void startRecorder() {
        Intent intent = new Intent(this, RecorderService.class).setAction(RecorderService.ACTION_START);
        startForegroundService(intent);
        getSharedPreferences("recorder", MODE_PRIVATE).edit().putBoolean("enabled", true).apply();
        status.setText("Status: nasłuchiwanie aktywne");
        Toast.makeText(this, "Nasłuchiwanie uruchomione", Toast.LENGTH_SHORT).show();
    }

    private void stopRecorder() {
        Intent intent = new Intent(this, RecorderService.class).setAction(RecorderService.ACTION_STOP);
        startService(intent);
        getSharedPreferences("recorder", MODE_PRIVATE).edit().putBoolean("enabled", false).apply();
        status.setText("Status: zatrzymany");
        Toast.makeText(this, "Nasłuchiwanie zatrzymane", Toast.LENGTH_SHORT).show();
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
