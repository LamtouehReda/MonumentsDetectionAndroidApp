package com.example.firstapp;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.text.Html;
import android.text.method.LinkMovementMethod;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
    Button photoBtn,galerieBtn,detectBtn;
    public static ImageView resultImage;
    TextView about;
    TextView dialog_language;
    String act2,act3;
    int lang_selected;

    private static final int IMAGE_PICK_CODE=1000;
    private static final int PERMISSION_CODE=1001;
    int imageSize=640;
    String imageString="";
   public static Bitmap image;
    OutputStream outputStream;
    Context context;
    Resources resources;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        photoBtn=findViewById(R.id.photoBtn);
        galerieBtn=findViewById(R.id.galerieBtn);
        dialog_language = (TextView)findViewById(R.id.dialog_language);



        photoBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                resultImage=(ImageView) findViewById(R.id.imageView8);

                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent,100);
            }
        });
        about=findViewById( R.id.about );
        about.setOnClickListener( new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent1=new Intent(MainActivity.this,MainActivity_about.class);
                intent1.putExtra( "key",act3 );
                startActivity(intent1);
            }
        } );

        galerieBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                resultImage=(ImageView) findViewById(R.id.imageView8);
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
                    if(checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE)==PackageManager.PERMISSION_DENIED){
                        String[] permissions={Manifest.permission.READ_EXTERNAL_STORAGE};
                        requestPermissions(permissions, PERMISSION_CODE);

                    }else{
                        pickImageFromGallery();
                    }
                }else{
                    pickImageFromGallery();
                } }
        });
        dialog_language.setText( "Fr" );
        act3="En";
        act2="Lancer la detection";
            if(LocaleHelper.getLanguage(MainActivity.this).equalsIgnoreCase("en"))
            {
                context = LocaleHelper.setLocale(MainActivity.this,"en");
                resources =context.getResources();
                dialog_language.setText("En");
                act3="En";
                lang_selected = 1;
            }else if(LocaleHelper.getLanguage(MainActivity.this).equalsIgnoreCase("fr")){
                context = LocaleHelper.setLocale(MainActivity.this,"fr");
                resources =context.getResources();
                dialog_language.setText("Fr");
                act3="Fr";
                lang_selected =0;
            }

        dialog_language.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    final String[] Language = {"Francais","English"};
                    final int checkItem;
                    Log.d("Clicked","Clicked");


                    final AlertDialog.Builder dialogBuilder = new AlertDialog.Builder(MainActivity.this);
                    dialogBuilder.setTitle("Select a Language")
                            .setSingleChoiceItems(Language, lang_selected, new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialogInterface, int i) {
                                    if(Language[i].equals("English")){
                                        dialog_language.setText("En");
                                        context = LocaleHelper.setLocale(MainActivity.this,"en");
                                        resources =context.getResources();
                                        photoBtn.setText(resources.getString(R.string.photo));
                                        galerieBtn.setText( resources.getString( R.string.gallery ) );
                                        about.setText( resources.getString( R.string.About ) );
                                        act2=resources.getString(R.string.detect);
                                        act3="En";
                                        lang_selected = 1;

                                    }
                                    if(Language[i].equals("Francais"))
                                    {   dialog_language.setText("Fr");
                                        context = LocaleHelper.setLocale(MainActivity.this,"fr");
                                        resources =context.getResources();
                                        photoBtn.setText(resources.getString(R.string.photo));
                                        galerieBtn.setText( resources.getString( R.string.gallery ) );
                                        about.setText( resources.getString( R.string.About ) );
                                        act2=resources.getString(R.string.detect);
                                        act3="Fr";
                                        lang_selected = 0;
                                    }

                                }
                            })
                            .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialogInterface, int i) {
                                    dialogInterface.dismiss();
                                }
                            });
                    dialogBuilder.create().show();
                }
            });
    }


    private void pickImageFromGallery() {
        //intent to pick image
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, IMAGE_PICK_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode){
            case PERMISSION_CODE:{
                if(grantResults.length > 0 && grantResults[0]==PackageManager.PERMISSION_GRANTED){
                    //PERMISSION WAS GRANTED
                    pickImageFromGallery();
                }else{
                    //permission was denied
                    Toast.makeText(this,"Permission denied...",Toast.LENGTH_SHORT).show();
                }
            }
        }
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 100) {
            //get capture Image
            assert data != null;
            image=(Bitmap) data.getExtras().get("data");
            Intent intent=new Intent(MainActivity.this,MainActivity_result.class);
            intent.putExtra( "lang",act2 );
            startActivity(intent);
        }
        if(resultCode == RESULT_OK && requestCode == IMAGE_PICK_CODE){
            assert data != null;
            Uri dat=data.getData();
            try {
                image= MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
            } catch (IOException e) {
                e.printStackTrace();
            }
            Intent intent=new Intent(MainActivity.this,MainActivity_result.class);
            intent.putExtra( "lang",act2 );
            startActivity(intent);
        }
    }

    public void saveImage(Bitmap image) {
        String fullPath = Environment.getExternalStorageDirectory().getAbsolutePath() + "/images/";
        try
        {
            File dir = new File(fullPath);
            if (!dir.exists()) {
                dir.mkdirs();
            }
            OutputStream fOut = null;
            File file = new File(fullPath, "image.png");
            if(file.exists())
                file.delete();
            file.createNewFile();
            fOut = new FileOutputStream(file);
            // 100 means no compression, the lower you go, the stronger the compression
            image.compress(Bitmap.CompressFormat.PNG, 100, fOut);
            fOut.flush();
            fOut.close();
        }
        catch (Exception e)
        {
            Log.e("saveToExternalStorage()", e.getMessage());
        }

    }
}