package com.example.firstapp;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;

public class MainActivity_about extends AppCompatActivity {
    String value;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate( savedInstanceState );
        setContentView( R.layout.activity_main_about );
        TextView text = findViewById( R.id.textView);
        Bundle extras = getIntent().getExtras();
        value = extras.getString( "key" );

            if(value.equals("Fr")){
            text.setText( "Cette application est un projet de fin d'études créé par Hamid Oufakir et Reda Lamtoueh\n" +
                    "sous l'encadrement du professeurs Sara Sekkate et Siham Akil, l'objectif principal de cette application est de détecter les monuments\n" +
                    "touristiques marocains célèbres et de fournir des informations sur ces monuments afin d'aider les touristes à reconnaître la culture marocaine.\n" +
                    "Cette application est capable de détecter 19 monuments marocains dans différentes villes." );

             }else{
            text.setText("This application is a graduation project created by Hamid Oufakir and Reda Lamtoueh\n" +
                    "under the supervision of professors Sara Sekkate and Siham Akil, the main objective of this application is to detect monuments\n" +
                    "famous Moroccan tourist attractions and provide information about these monuments to help tourists recognize Moroccan culture. This application is able to detect 19 Moroccan monuments in different cities.");

                    }


    }
}