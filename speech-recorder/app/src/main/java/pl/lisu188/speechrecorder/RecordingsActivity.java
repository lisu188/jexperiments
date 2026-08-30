package pl.lisu188.speechrecorder;

import android.app.Activity;
import android.content.ContentUris;
import android.database.Cursor;
import android.media.MediaMetadataRetriever;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Gravity;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class RecordingsActivity extends Activity {
    private final List<Recording> recordings = new ArrayList<>();
    private final List<String> labels = new ArrayList<>();
    private ArrayAdapter<String> adapter;
    private MediaPlayer player;
    private Uri playingUri;
    private TextView nowPlaying;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(buildUi());
        loadRecordings();
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadRecordings();
    }

    @Override
    protected void onDestroy() {
        stopPlayback();
        super.onDestroy();
    }

    private View buildUi() {
        int pad = dp(20);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);

        TextView title = new TextView(this);
        title.setText("Nagrania");
        title.setTextSize(26);
        title.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(title, matchWrap());

        TextView hint = new TextView(this);
        hint.setText("Dotknij nagrania, aby je odtworzyć lub zatrzymać.");
        hint.setTextSize(14);
        hint.setPadding(0, dp(10), 0, dp(14));
        root.addView(hint, matchWrap());

        nowPlaying = new TextView(this);
        nowPlaying.setText("Nic nie jest odtwarzane");
        nowPlaying.setTextSize(15);
        nowPlaying.setPadding(0, 0, 0, dp(10));
        root.addView(nowPlaying, matchWrap());

        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);

        Button refresh = new Button(this);
        refresh.setText("ODŚWIEŻ");
        refresh.setOnClickListener(v -> loadRecordings());
        actions.addView(refresh, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        Button stop = new Button(this);
        stop.setText("STOP");
        stop.setOnClickListener(v -> stopPlayback());
        LinearLayout.LayoutParams stopParams = new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
        stopParams.leftMargin = dp(8);
        actions.addView(stop, stopParams);
        root.addView(actions, matchWrap());

        ListView list = new ListView(this);
        adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, labels);
        list.setAdapter(adapter);
        list.setOnItemClickListener((parent, view, position, id) -> {
            if (recordings.isEmpty()) {
                return;
            }
            togglePlayback(recordings.get(position));
        });
        LinearLayout.LayoutParams listParams = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f);
        listParams.topMargin = dp(10);
        root.addView(list, listParams);

        return root;
    }

    private void loadRecordings() {
        recordings.clear();
        labels.clear();

        Uri collection = MediaStore.Audio.Media.EXTERNAL_CONTENT_URI;
        String[] projection = {
                MediaStore.Audio.Media._ID,
                MediaStore.Audio.Media.DISPLAY_NAME,
                MediaStore.Audio.Media.DATE_ADDED,
                MediaStore.Audio.Media.SIZE,
                MediaStore.Audio.Media.DURATION
        };
        String selection = MediaStore.Audio.Media.RELATIVE_PATH + "=?";
        String[] args = {"Music/SpeechRecorder/"};
        String sort = MediaStore.Audio.Media.DATE_ADDED + " DESC";

        try (Cursor cursor = getContentResolver().query(collection, projection, selection, args, sort)) {
            if (cursor != null) {
                int idColumn = cursor.getColumnIndexOrThrow(MediaStore.Audio.Media._ID);
                int nameColumn = cursor.getColumnIndexOrThrow(MediaStore.Audio.Media.DISPLAY_NAME);
                int dateColumn = cursor.getColumnIndexOrThrow(MediaStore.Audio.Media.DATE_ADDED);
                int sizeColumn = cursor.getColumnIndexOrThrow(MediaStore.Audio.Media.SIZE);
                int durationColumn = cursor.getColumnIndexOrThrow(MediaStore.Audio.Media.DURATION);

                while (cursor.moveToNext()) {
                    long id = cursor.getLong(idColumn);
                    String name = cursor.getString(nameColumn);
                    long dateAdded = cursor.getLong(dateColumn) * 1000L;
                    long size = cursor.getLong(sizeColumn);
                    long duration = cursor.getLong(durationColumn);
                    Uri uri = ContentUris.withAppendedId(collection, id);
                    if (duration <= 0) {
                        duration = readDuration(uri);
                    }
                    Recording recording = new Recording(uri, name, dateAdded, size, duration);
                    recordings.add(recording);
                    labels.add(formatLabel(recording));
                }
            }
        } catch (Exception e) {
            Toast.makeText(this, "Nie udało się odczytać nagrań", Toast.LENGTH_LONG).show();
        }

        if (labels.isEmpty()) {
            labels.add("Brak nagrań w Music/SpeechRecorder");
        }
        adapter.notifyDataSetChanged();
    }

    private long readDuration(Uri uri) {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        try {
            retriever.setDataSource(this, uri);
            String value = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            return value == null ? 0 : Long.parseLong(value);
        } catch (Exception ignored) {
            return 0;
        } finally {
            try {
                retriever.release();
            } catch (Exception ignored) {
            }
        }
    }

    private String formatLabel(Recording recording) {
        String date = DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.MEDIUM).format(new Date(recording.dateAdded));
        return recording.name + "\n" + date + "   •   " + formatDuration(recording.durationMs) + "   •   " + formatSize(recording.sizeBytes);
    }

    private String formatDuration(long ms) {
        long totalSeconds = Math.max(0, ms / 1000L);
        long minutes = totalSeconds / 60;
        long seconds = totalSeconds % 60;
        return String.format(Locale.getDefault(), "%d:%02d", minutes, seconds);
    }

    private String formatSize(long bytes) {
        if (bytes < 1024 * 1024) {
            return String.format(Locale.getDefault(), "%.1f KB", bytes / 1024.0);
        }
        return String.format(Locale.getDefault(), "%.1f MB", bytes / (1024.0 * 1024.0));
    }

    private void togglePlayback(Recording recording) {
        if (playingUri != null && playingUri.equals(recording.uri) && player != null && player.isPlaying()) {
            stopPlayback();
            return;
        }

        stopPlayback();
        try {
            player = new MediaPlayer();
            player.setDataSource(this, recording.uri);
            player.setOnCompletionListener(mp -> stopPlayback());
            player.prepare();
            player.start();
            playingUri = recording.uri;
            nowPlaying.setText("Odtwarzanie: " + recording.name);
        } catch (Exception e) {
            stopPlayback();
            Toast.makeText(this, "Nie udało się odtworzyć nagrania", Toast.LENGTH_LONG).show();
        }
    }

    private void stopPlayback() {
        if (player != null) {
            try {
                player.stop();
            } catch (Exception ignored) {
            }
            try {
                player.release();
            } catch (Exception ignored) {
            }
            player = null;
        }
        playingUri = null;
        if (nowPlaying != null) {
            nowPlaying.setText("Nic nie jest odtwarzane");
        }
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private static final class Recording {
        final Uri uri;
        final String name;
        final long dateAdded;
        final long sizeBytes;
        final long durationMs;

        Recording(Uri uri, String name, long dateAdded, long sizeBytes, long durationMs) {
            this.uri = uri;
            this.name = name;
            this.dateAdded = dateAdded;
            this.sizeBytes = sizeBytes;
            this.durationMs = durationMs;
        }
    }
}
