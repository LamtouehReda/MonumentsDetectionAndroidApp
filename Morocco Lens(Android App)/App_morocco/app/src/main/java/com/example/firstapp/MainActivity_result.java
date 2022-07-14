package com.example.firstapp;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.Html;
import android.text.method.LinkMovementMethod;
import android.util.Base64;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class MainActivity_result extends AppCompatActivity {
    Button detectBtn;
    ImageView resultImage;
    TextView resultText,monument;
    private static final int IMAGE_PICK_CODE=1000;
    private static final int PERMISSION_CODE=1001;
    int imageSize=640;
    String imageString="";
    Bitmap image;
    String value;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate( savedInstanceState );
        setContentView( R.layout.activity_main_result );
        Bundle extras = getIntent().getExtras();
        if (extras != null) {
            value = extras.getString( "lang" );
        }
        resultImage=(ImageView)findViewById(R.id.result);
        resultImage.setImageBitmap(MainActivity.image);
        detectBtn=findViewById( R.id.detectBtn );
        resultText=findViewById( R.id.resultText );
        detectBtn.setText( value );
        monument=findViewById( R.id.monument );
        detectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                classifyImage(MainActivity.image);
            }
        });

    }
    public void classifyImage(Bitmap image){
        if (! Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
        detectBtn.setVisibility(View.GONE);
        imageString=getStringImage(image);
        Python py=Python.getInstance();
        PyObject pyobj=py.getModule("detect");
        PyObject obj=pyobj.callAttr("main",imageString);
        PyObject labelObj=pyobj.callAttr("MonumentLabel");
        PyObject monName=pyobj.callAttr("monumentName");
        String str=obj.toString();
        String label=labelObj.toString();
        String mon_name=monName.toString();
        byte data[]=android.util.Base64.decode(str, Base64.DEFAULT);
        Bitmap bmp= BitmapFactory.decodeByteArray(data,0,data.length);
        resultImage.setImageBitmap(bmp);
        monument.setText(mon_name);
        String urlMon=mon_name.replace(' ','+');;
        String dynamicUrl = "https://www.google.com/search?q="+urlMon+"&oq=chrome.2.69i57j0i512l2j46i175i199i512l3j0i512j46i175i199i512j46i10i175i199i512j46i175i199i512.3974j0j9&sourceid=chrome&ie=UTF-8";
        String linkedText =label+" "+String.format("<a href=\"%s\">Plus D'informations...</a>", dynamicUrl);
        resultText.setText( Html.fromHtml(linkedText));
        resultText.setMovementMethod( LinkMovementMethod.getInstance());
        resultImage.getLayoutParams().width = 650;
        resultText.getLayoutParams().height = 254;
        resultText.setVisibility( View.VISIBLE );
        monument.setVisibility( View.VISIBLE );
        detectBtn.setVisibility(View.GONE);
    }
    private String getStringImage(Bitmap image) {
        ByteArrayOutputStream baos=new ByteArrayOutputStream();
        image.compress(Bitmap.CompressFormat.JPEG,100,baos);
        byte[] imageBytes = baos.toByteArray();
        String stringEncodedImage=android.util.Base64.encodeToString(imageBytes, Base64.DEFAULT);
        return  stringEncodedImage;
    }
}