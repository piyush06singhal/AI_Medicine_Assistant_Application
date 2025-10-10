"""
Advanced AI Medical Assistant - Built-in Medical Knowledge Base
No external APIs required - Fast, reliable, and accurate medical insights
"""

import streamlit as st
from datetime import datetime
from PIL import Image
import io
import re

# Page configuration
st.set_page_config(
    page_title="🏥 Advanced AI Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Comprehensive Medical Knowledge Base
MEDICAL_DATABASE = {
    'diabetes': {
        'keywords': ['diabetes', 'sugar', 'glucose', 'insulin', 'urination', 'thirst', 'thirsty', 'frequent urination', 'excessive thirst', 'weight loss', 'hunger', 'blurred vision', 'fatigue', 'blood sugar'],
        'description': 'Diabetes is a chronic condition that affects how your body processes blood sugar (glucose).',
        'symptoms': [
            'Increased thirst and frequent urination',
            'Extreme hunger',
            'Unexplained weight loss',
            'Fatigue and weakness',
            'Blurred vision',
            'Slow-healing sores or frequent infections',
            'Tingling or numbness in hands or feet'
        ],
        'precautions': [
            'Monitor blood sugar levels regularly',
            'Follow a balanced diet with controlled carbohydrates',
            'Exercise regularly (at least 30 minutes daily)',
            'Take prescribed medications as directed',
            'Maintain a healthy weight',
            'Get regular check-ups with your healthcare provider',
            'Check your feet daily for cuts or blisters',
            'Manage stress through relaxation techniques'
        ],
        'when_to_see_doctor': 'Seek immediate medical attention if you experience extreme thirst, frequent urination, unexplained weight loss, or blood sugar levels consistently above 240 mg/dL.',
        'severity': 'High',
        'confidence': 0.85
    },
    'hypertension': {
        'keywords': ['hypertension', 'high blood pressure', 'bp', 'blood pressure', 'headache', 'dizziness', 'chest pain', 'shortness of breath', 'nosebleeds', 'pressure'],
        'description': 'Hypertension (high blood pressure) is a condition where the force of blood against artery walls is consistently too high.',
        'symptoms': [
            'Severe headaches',
            'Shortness of breath',
            'Nosebleeds',
            'Dizziness',
            'Chest pain',
            'Visual changes',
            'Fatigue',
            'Irregular heartbeat'
        ],
        'precautions': [
            'Reduce sodium intake (less than 2,300mg daily)',
            'Exercise regularly (150 minutes per week)',
            'Maintain a healthy weight',
            'Limit alcohol consumption',
            'Quit smoking',
            'Manage stress through meditation or yoga',
            'Take blood pressure medications as prescribed',
            'Monitor blood pressure regularly at home'
        ],
        'when_to_see_doctor': 'Seek emergency care if blood pressure is 180/120 or higher, or if you experience severe headache, chest pain, or difficulty breathing.',
        'severity': 'High',
        'confidence': 0.82
    },
    'migraine': {
        'keywords': ['migraine', 'headache', 'severe headache', 'throbbing', 'pounding', 'nausea', 'light sensitivity', 'sound sensitivity', 'aura', 'visual disturbances'],
        'description': 'Migraine is a neurological condition characterized by intense, debilitating headaches often accompanied by other symptoms.',
        'symptoms': [
            'Intense throbbing or pulsing pain (usually on one side)',
            'Nausea and vomiting',
            'Sensitivity to light and sound',
            'Visual disturbances (aura)',
            'Dizziness',
            'Neck stiffness',
            'Difficulty concentrating'
        ],
        'precautions': [
            'Identify and avoid triggers (stress, certain foods, lack of sleep)',
            'Maintain a regular sleep schedule',
            'Stay hydrated (8-10 glasses of water daily)',
            'Practice stress management techniques',
            'Take preventive medications if prescribed',
            'Keep a migraine diary to track patterns',
            'Apply cold or warm compresses to head/neck',
            'Rest in a quiet, dark room during attacks'
        ],
        'when_to_see_doctor': 'Seek immediate care if headache is sudden and severe, accompanied by fever, stiff neck, confusion, vision problems, or difficulty speaking.',
        'severity': 'Moderate',
        'confidence': 0.78
    },
    'flu': {
        'keywords': ['flu', 'influenza', 'fever', 'cough', 'sore throat', 'body aches', 'muscle aches', 'chills', 'fatigue', 'headache', 'runny nose', 'stuffy nose'],
        'description': 'Influenza (flu) is a contagious respiratory illness caused by influenza viruses.',
        'symptoms': [
            'Fever (usually high)',
            'Cough',
            'Sore throat',
            'Runny or stuffy nose',
            'Body aches and muscle pain',
            'Headaches',
            'Chills and sweats',
            'Fatigue and weakness'
        ],
        'precautions': [
            'Get plenty of rest (7-9 hours nightly)',
            'Stay hydrated with water, warm liquids, and broths',
            'Take over-the-counter medications for symptom relief',
            'Avoid close contact with others to prevent spread',
            'Cover coughs and sneezes',
            'Wash hands frequently',
            'Get annual flu vaccination',
            'Stay home from work or school until fever-free for 24 hours'
        ],
        'when_to_see_doctor': 'Seek medical care if you have difficulty breathing, chest pain, persistent fever above 103°F, severe weakness, or symptoms that improve then worsen.',
        'severity': 'Moderate',
        'confidence': 0.80
    },
    'pneumonia': {
        'keywords': ['pneumonia', 'lung infection', 'chest pain', 'cough', 'fever', 'shortness of breath', 'breathing difficulty', 'chills', 'phlegm', 'respiratory'],
        'description': 'Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid.',
        'symptoms': [
            'Chest pain when breathing or coughing',
            'Cough with phlegm or pus',
            'Fever, sweating, and chills',
            'Shortness of breath',
            'Fatigue and weakness',
            'Nausea, vomiting, or diarrhea',
            'Confusion (especially in older adults)'
        ],
        'precautions': [
            'Take prescribed antibiotics as directed',
            'Get plenty of rest',
            'Stay hydrated (8-10 glasses daily)',
            'Use a humidifier to ease breathing',
            'Avoid smoking and secondhand smoke',
            'Practice good hand hygiene',
            'Get pneumonia and flu vaccines',
            'Follow up with healthcare provider regularly'
        ],
        'when_to_see_doctor': 'Seek emergency care if you have severe difficulty breathing, chest pain, persistent high fever, or bluish lips/fingernails.',
        'severity': 'High',
        'confidence': 0.83
    },
    'asthma': {
        'keywords': ['asthma', 'wheezing', 'shortness of breath', 'chest tightness', 'coughing', 'breathing difficulty', 'respiratory', 'bronchial'],
        'description': 'Asthma is a chronic condition where airways narrow and swell, producing extra mucus and making breathing difficult.',
        'symptoms': [
            'Wheezing (whistling sound when breathing)',
            'Shortness of breath',
            'Chest tightness or pain',
            'Coughing (especially at night or early morning)',
            'Difficulty sleeping due to breathing problems',
            'Rapid breathing',
            'Fatigue during physical activity'
        ],
        'precautions': [
            'Use prescribed inhalers as directed',
            'Avoid known triggers (allergens, smoke, pollution)',
            'Monitor peak flow regularly',
            'Create and follow an asthma action plan',
            'Get annual flu and pneumonia vaccines',
            'Maintain good indoor air quality',
            'Exercise regularly with proper warm-up',
            'Keep rescue inhaler always accessible'
        ],
        'when_to_see_doctor': 'Seek emergency care if breathing becomes very difficult, lips/fingernails turn blue, or rescue inhaler doesn\'t help.',
        'severity': 'High',
        'confidence': 0.81
    },
    'anxiety': {
        'keywords': ['anxiety', 'worry', 'nervous', 'panic', 'stress', 'restlessness', 'fear', 'apprehension', 'tension', 'irritability', 'sleep problems', 'concentration'],
        'description': 'Anxiety is a mental health condition characterized by excessive worry, fear, and nervousness.',
        'symptoms': [
            'Excessive worry or fear',
            'Restlessness or feeling on edge',
            'Rapid heart rate',
            'Sweating and trembling',
            'Difficulty concentrating',
            'Sleep problems',
            'Irritability',
            'Muscle tension'
        ],
        'precautions': [
            'Practice relaxation techniques (deep breathing, meditation)',
            'Exercise regularly (30 minutes daily)',
            'Get adequate sleep (7-9 hours)',
            'Limit caffeine and alcohol',
            'Consider therapy or counseling',
            'Maintain a regular routine',
            'Connect with supportive friends and family',
            'Practice mindfulness and stress management'
        ],
        'when_to_see_doctor': 'Seek help if anxiety interferes with daily life, causes panic attacks, or leads to thoughts of self-harm.',
        'severity': 'Moderate',
        'confidence': 0.75
    },
    'depression': {
        'keywords': ['depression', 'sadness', 'hopelessness', 'mood', 'emotional', 'fatigue', 'sleep problems', 'appetite changes', 'concentration', 'worthless', 'guilt'],
        'description': 'Depression is a mood disorder causing persistent feelings of sadness and loss of interest.',
        'symptoms': [
            'Persistent sadness or hopelessness',
            'Loss of interest in activities',
            'Fatigue and decreased energy',
            'Sleep disturbances (too much or too little)',
            'Appetite or weight changes',
            'Difficulty concentrating',
            'Feelings of worthlessness or guilt',
            'Thoughts of death or suicide'
        ],
        'precautions': [
            'Seek professional help (therapy, counseling)',
            'Take prescribed medications as directed',
            'Maintain regular sleep schedule',
            'Exercise regularly',
            'Stay connected with friends and family',
            'Avoid alcohol and drugs',
            'Practice stress management',
            'Set realistic goals and priorities'
        ],
        'when_to_see_doctor': 'Seek immediate help if you have thoughts of suicide or self-harm. Call emergency services or a crisis hotline.',
        'severity': 'High',
        'confidence': 0.77
    },
    'common_cold': {
        'keywords': ['cold', 'runny nose', 'stuffy nose', 'sneezing', 'sore throat', 'cough', 'congestion', 'mild fever'],
        'description': 'The common cold is a viral infection of the upper respiratory tract.',
        'symptoms': [
            'Runny or stuffy nose',
            'Sneezing',
            'Sore throat',
            'Cough',
            'Mild headache',
            'Low-grade fever',
            'Fatigue',
            'Watery eyes'
        ],
        'precautions': [
            'Get plenty of rest',
            'Stay hydrated with water and warm liquids',
            'Use saline nasal drops or spray',
            'Gargle with salt water for sore throat',
            'Take over-the-counter pain relievers if needed',
            'Use a humidifier',
            'Wash hands frequently',
            'Avoid close contact with others'
        ],
        'when_to_see_doctor': 'See a doctor if symptoms last more than 10 days, fever is above 101.3°F, or you have severe symptoms.',
        'severity': 'Low',
        'confidence': 0.85
    },
    'allergies': {
        'keywords': ['allergy', 'allergies', 'sneezing', 'itchy', 'watery eyes', 'runny nose', 'rash', 'hives', 'itching', 'congestion'],
        'description': 'Allergies occur when your immune system reacts to a foreign substance.',
        'symptoms': [
            'Sneezing',
            'Itchy, watery eyes',
            'Runny or stuffy nose',
            'Itchy throat or ears',
            'Skin rash or hives',
            'Coughing',
            'Fatigue',
            'Headache'
        ],
        'precautions': [
            'Identify and avoid allergens',
            'Take antihistamines as needed',
            'Use air purifiers indoors',
            'Keep windows closed during high pollen days',
            'Shower after being outdoors',
            'Wash bedding regularly in hot water',
            'Consider allergy shots (immunotherapy)',
            'Use nasal saline rinses'
        ],
        'when_to_see_doctor': 'Seek emergency care if you experience difficulty breathing, swelling of face/throat, or signs of anaphylaxis.',
        'severity': 'Low to Moderate',
        'confidence': 0.80
    },
    'bronchitis': {
        'keywords': ['bronchitis', 'chest congestion', 'mucus', 'persistent cough', 'wheezing', 'chest discomfort', 'phlegm'],
        'description': 'Bronchitis is inflammation of the bronchial tubes that carry air to your lungs.',
        'symptoms': [
            'Persistent cough with mucus',
            'Chest discomfort',
            'Fatigue',
            'Shortness of breath',
            'Slight fever and chills',
            'Wheezing'
        ],
        'precautions': [
            'Get plenty of rest',
            'Drink lots of fluids',
            'Use a humidifier',
            'Avoid lung irritants',
            'Take cough medicine if needed',
            'Quit smoking'
        ],
        'when_to_see_doctor': 'See a doctor if cough lasts more than 3 weeks, prevents sleep, or you cough up blood.',
        'severity': 'Moderate',
        'confidence': 0.79
    },
    'gastritis': {
        'keywords': ['gastritis', 'stomach pain', 'indigestion', 'nausea', 'vomiting', 'bloating', 'burning stomach', 'upper abdominal pain'],
        'description': 'Gastritis is inflammation of the stomach lining.',
        'symptoms': [
            'Burning or gnawing pain in upper abdomen',
            'Nausea and vomiting',
            'Feeling of fullness',
            'Loss of appetite',
            'Bloating',
            'Hiccups'
        ],
        'precautions': [
            'Eat smaller, more frequent meals',
            'Avoid spicy and acidic foods',
            'Limit alcohol consumption',
            'Manage stress',
            'Avoid NSAIDs (like ibuprofen)',
            'Take antacids as needed'
        ],
        'when_to_see_doctor': 'Seek care if you vomit blood, have black stools, or severe abdominal pain.',
        'severity': 'Moderate',
        'confidence': 0.76
    },
    'arthritis': {
        'keywords': ['arthritis', 'joint pain', 'stiffness', 'swelling', 'inflammation', 'joint stiffness', 'rheumatoid', 'osteoarthritis'],
        'description': 'Arthritis is inflammation of one or more joints causing pain and stiffness.',
        'symptoms': [
            'Joint pain and tenderness',
            'Stiffness (especially in morning)',
            'Swelling around joints',
            'Reduced range of motion',
            'Redness and warmth',
            'Fatigue'
        ],
        'precautions': [
            'Maintain healthy weight',
            'Exercise regularly (low-impact)',
            'Apply hot or cold therapy',
            'Take anti-inflammatory medications',
            'Use assistive devices if needed',
            'Physical therapy',
            'Protect joints from overuse'
        ],
        'when_to_see_doctor': 'See a doctor if joint pain is severe, sudden, or accompanied by fever.',
        'severity': 'Moderate to High',
        'confidence': 0.81
    },
    'uti': {
        'keywords': ['uti', 'urinary tract infection', 'burning urination', 'frequent urination', 'bladder infection', 'painful urination', 'pelvic pain'],
        'description': 'UTI is an infection in any part of the urinary system.',
        'symptoms': [
            'Strong, persistent urge to urinate',
            'Burning sensation when urinating',
            'Frequent, small amounts of urine',
            'Cloudy or strong-smelling urine',
            'Pelvic pain',
            'Blood in urine'
        ],
        'precautions': [
            'Drink plenty of water',
            'Urinate frequently',
            'Wipe front to back',
            'Empty bladder after intercourse',
            'Avoid irritating feminine products',
            'Take prescribed antibiotics completely'
        ],
        'when_to_see_doctor': 'See a doctor if you have fever, back pain, nausea, or symptoms persist.',
        'severity': 'Moderate',
        'confidence': 0.84
    },
    'sinusitis': {
        'keywords': ['sinusitis', 'sinus infection', 'facial pain', 'nasal congestion', 'thick nasal discharge', 'sinus pressure', 'postnasal drip'],
        'description': 'Sinusitis is inflammation or swelling of the tissue lining the sinuses.',
        'symptoms': [
            'Facial pain and pressure',
            'Nasal congestion',
            'Thick yellow or green discharge',
            'Reduced sense of smell',
            'Cough',
            'Headache',
            'Fatigue'
        ],
        'precautions': [
            'Use saline nasal spray',
            'Apply warm compresses',
            'Stay hydrated',
            'Use a humidifier',
            'Avoid allergens and irritants',
            'Take decongestants if needed'
        ],
        'when_to_see_doctor': 'See a doctor if symptoms last more than 10 days or worsen after initial improvement.',
        'severity': 'Low to Moderate',
        'confidence': 0.77
    },
    'eczema': {
        'keywords': ['eczema', 'atopic dermatitis', 'itchy skin', 'dry skin', 'rash', 'red patches', 'skin inflammation'],
        'description': 'Eczema is a condition that makes skin red, inflamed, and itchy.',
        'symptoms': [
            'Dry, sensitive skin',
            'Intense itching',
            'Red or brownish patches',
            'Small, raised bumps',
            'Thickened, cracked skin',
            'Raw, swollen skin from scratching'
        ],
        'precautions': [
            'Moisturize skin regularly',
            'Avoid harsh soaps and detergents',
            'Take lukewarm baths',
            'Use gentle, fragrance-free products',
            'Avoid scratching',
            'Identify and avoid triggers',
            'Use prescribed topical medications'
        ],
        'when_to_see_doctor': 'See a doctor if eczema interferes with sleep, shows signs of infection, or doesn\'t improve.',
        'severity': 'Low to Moderate',
        'confidence': 0.73
    },
    'gerd': {
        'keywords': ['gerd', 'acid reflux', 'heartburn', 'chest burning', 'regurgitation', 'sour taste', 'difficulty swallowing'],
        'description': 'GERD is a chronic condition where stomach acid flows back into the esophagus.',
        'symptoms': [
            'Burning sensation in chest (heartburn)',
            'Regurgitation of food or sour liquid',
            'Difficulty swallowing',
            'Chest pain',
            'Chronic cough',
            'Sore throat',
            'Feeling of lump in throat'
        ],
        'precautions': [
            'Avoid trigger foods (spicy, fatty, acidic)',
            'Eat smaller meals',
            'Don\'t lie down after eating',
            'Elevate head of bed',
            'Maintain healthy weight',
            'Quit smoking',
            'Take antacids or prescribed medications'
        ],
        'when_to_see_doctor': 'Seek care if you have severe chest pain, difficulty swallowing, or persistent vomiting.',
        'severity': 'Moderate',
        'confidence': 0.80
    },
    'thyroid_disorder': {
        'keywords': ['thyroid', 'hypothyroid', 'hyperthyroid', 'weight changes', 'fatigue', 'metabolism', 'goiter', 'thyroid gland'],
        'description': 'Thyroid disorders affect the thyroid gland which regulates metabolism.',
        'symptoms': [
            'Unexplained weight changes',
            'Fatigue or nervousness',
            'Changes in heart rate',
            'Sensitivity to temperature',
            'Changes in bowel patterns',
            'Muscle weakness',
            'Mood changes'
        ],
        'precautions': [
            'Take thyroid medication as prescribed',
            'Get regular blood tests',
            'Maintain balanced diet',
            'Manage stress',
            'Get adequate sleep',
            'Exercise regularly'
        ],
        'when_to_see_doctor': 'See a doctor if you notice persistent symptoms or rapid heart rate.',
        'severity': 'Moderate to High',
        'confidence': 0.78
    },
    'anemia': {
        'keywords': ['anemia', 'fatigue', 'weakness', 'pale skin', 'shortness of breath', 'dizziness', 'cold hands', 'iron deficiency'],
        'description': 'Anemia is a condition where you lack enough healthy red blood cells.',
        'symptoms': [
            'Fatigue and weakness',
            'Pale or yellowish skin',
            'Shortness of breath',
            'Dizziness or lightheadedness',
            'Cold hands and feet',
            'Chest pain',
            'Headaches'
        ],
        'precautions': [
            'Eat iron-rich foods',
            'Take iron supplements if prescribed',
            'Consume vitamin C to aid iron absorption',
            'Treat underlying causes',
            'Get regular blood tests',
            'Avoid excessive tea and coffee'
        ],
        'when_to_see_doctor': 'See a doctor if you have persistent fatigue, rapid heartbeat, or pale skin.',
        'severity': 'Moderate',
        'confidence': 0.79
    },
    'insomnia': {
        'keywords': ['insomnia', 'sleep problems', 'can\'t sleep', 'difficulty sleeping', 'sleeplessness', 'trouble falling asleep', 'waking up'],
        'description': 'Insomnia is a sleep disorder that makes it hard to fall or stay asleep.',
        'symptoms': [
            'Difficulty falling asleep',
            'Waking up during the night',
            'Waking up too early',
            'Daytime tiredness',
            'Irritability or depression',
            'Difficulty concentrating',
            'Increased errors or accidents'
        ],
        'precautions': [
            'Stick to a sleep schedule',
            'Create a restful environment',
            'Limit daytime naps',
            'Exercise regularly',
            'Avoid caffeine and alcohol before bed',
            'Manage stress and worries',
            'Avoid screens before bedtime'
        ],
        'when_to_see_doctor': 'See a doctor if insomnia persists for more than a few weeks or affects daily life.',
        'severity': 'Low to Moderate',
        'confidence': 0.74
    },
    'constipation': {
        'keywords': ['constipation', 'bowel movement', 'hard stools', 'difficulty passing stool', 'bloating', 'abdominal discomfort'],
        'description': 'Constipation is infrequent bowel movements or difficulty passing stools.',
        'symptoms': [
            'Fewer than three bowel movements per week',
            'Hard or lumpy stools',
            'Straining during bowel movements',
            'Feeling of incomplete evacuation',
            'Abdominal bloating',
            'Abdominal pain'
        ],
        'precautions': [
            'Increase fiber intake',
            'Drink plenty of water',
            'Exercise regularly',
            'Don\'t ignore urge to have bowel movement',
            'Establish regular bathroom routine',
            'Consider fiber supplements or laxatives if needed'
        ],
        'when_to_see_doctor': 'See a doctor if constipation is severe, lasts more than 3 weeks, or you have blood in stool.',
        'severity': 'Low',
        'confidence': 0.82
    },
    'diarrhea': {
        'keywords': ['diarrhea', 'loose stools', 'watery stools', 'frequent bowel movements', 'stomach cramps', 'dehydration'],
        'description': 'Diarrhea is loose, watery stools occurring more frequently than normal.',
        'symptoms': [
            'Loose, watery stools',
            'Abdominal cramps',
            'Abdominal pain',
            'Fever',
            'Bloating',
            'Nausea',
            'Urgent need to have bowel movement'
        ],
        'precautions': [
            'Stay hydrated with water and electrolytes',
            'Eat bland foods (BRAT diet)',
            'Avoid dairy, fatty, and spicy foods',
            'Rest',
            'Take anti-diarrheal medication if needed',
            'Practice good hand hygiene'
        ],
        'when_to_see_doctor': 'Seek care if diarrhea lasts more than 2 days, you have severe pain, or signs of dehydration.',
        'severity': 'Low to Moderate',
        'confidence': 0.83
    },
    'chickenpox': {
        'keywords': ['chickenpox', 'varicella', 'itchy rash', 'blisters', 'fever', 'spots', 'pox'],
        'description': 'Chickenpox is a highly contagious viral infection causing an itchy rash.',
        'symptoms': [
            'Itchy rash with red spots',
            'Fluid-filled blisters',
            'Fever',
            'Fatigue',
            'Loss of appetite',
            'Headache'
        ],
        'precautions': [
            'Isolate to prevent spread',
            'Apply calamine lotion for itching',
            'Take lukewarm baths with oatmeal',
            'Avoid scratching',
            'Take fever reducers (avoid aspirin)',
            'Stay hydrated',
            'Get vaccinated for prevention'
        ],
        'when_to_see_doctor': 'See a doctor if you have difficulty breathing, severe headache, or rash spreads to eyes.',
        'severity': 'Moderate',
        'confidence': 0.86
    },
    'measles': {
        'keywords': ['measles', 'rash', 'fever', 'cough', 'runny nose', 'red eyes', 'koplik spots'],
        'description': 'Measles is a highly contagious viral infection.',
        'symptoms': [
            'High fever',
            'Cough',
            'Runny nose',
            'Red, watery eyes',
            'Tiny white spots in mouth',
            'Red rash that spreads'
        ],
        'precautions': [
            'Get MMR vaccine',
            'Isolate to prevent spread',
            'Rest and stay hydrated',
            'Take fever reducers',
            'Avoid bright lights if eyes are sensitive',
            'Boost immune system with vitamin A'
        ],
        'when_to_see_doctor': 'Seek immediate care for difficulty breathing, severe headache, or confusion.',
        'severity': 'High',
        'confidence': 0.85
    },
    'mumps': {
        'keywords': ['mumps', 'swollen glands', 'parotid glands', 'jaw pain', 'fever', 'swollen cheeks'],
        'description': 'Mumps is a viral infection affecting the salivary glands.',
        'symptoms': [
            'Swollen, painful salivary glands',
            'Fever',
            'Headache',
            'Muscle aches',
            'Fatigue',
            'Loss of appetite',
            'Pain while chewing or swallowing'
        ],
        'precautions': [
            'Get MMR vaccine',
            'Rest and isolate',
            'Apply warm or cold compresses',
            'Eat soft foods',
            'Stay hydrated',
            'Take pain relievers'
        ],
        'when_to_see_doctor': 'See a doctor if you have severe headache, stiff neck, or testicular pain.',
        'severity': 'Moderate',
        'confidence': 0.82
    },
    'dengue': {
        'keywords': ['dengue', 'dengue fever', 'high fever', 'severe headache', 'pain behind eyes', 'joint pain', 'rash', 'mosquito'],
        'description': 'Dengue is a mosquito-borne viral infection.',
        'symptoms': [
            'High fever',
            'Severe headache',
            'Pain behind the eyes',
            'Joint and muscle pain',
            'Nausea and vomiting',
            'Skin rash',
            'Mild bleeding'
        ],
        'precautions': [
            'Rest and stay hydrated',
            'Take acetaminophen for fever (avoid aspirin)',
            'Use mosquito repellent',
            'Wear protective clothing',
            'Eliminate standing water',
            'Monitor for warning signs'
        ],
        'when_to_see_doctor': 'Seek emergency care for severe abdominal pain, persistent vomiting, or bleeding.',
        'severity': 'High',
        'confidence': 0.84
    },
    'malaria': {
        'keywords': ['malaria', 'fever', 'chills', 'sweating', 'headache', 'mosquito', 'shivering', 'cyclical fever'],
        'description': 'Malaria is a life-threatening disease transmitted by mosquitoes.',
        'symptoms': [
            'Cyclical fever and chills',
            'Sweating',
            'Headache',
            'Nausea and vomiting',
            'Muscle pain',
            'Fatigue',
            'Diarrhea'
        ],
        'precautions': [
            'Take antimalarial medications as prescribed',
            'Use mosquito nets',
            'Apply insect repellent',
            'Wear long sleeves and pants',
            'Stay in screened areas',
            'Seek immediate treatment'
        ],
        'when_to_see_doctor': 'Seek immediate medical care if you have fever after visiting malaria-endemic areas.',
        'severity': 'High',
        'confidence': 0.87
    },
    'tuberculosis': {
        'keywords': ['tuberculosis', 'tb', 'persistent cough', 'coughing blood', 'chest pain', 'weight loss', 'night sweats', 'lung infection'],
        'description': 'Tuberculosis is a bacterial infection that primarily affects the lungs.',
        'symptoms': [
            'Persistent cough (3+ weeks)',
            'Coughing up blood',
            'Chest pain',
            'Unintentional weight loss',
            'Fatigue',
            'Fever and night sweats',
            'Loss of appetite'
        ],
        'precautions': [
            'Complete full course of antibiotics',
            'Isolate during infectious period',
            'Cover mouth when coughing',
            'Ensure good ventilation',
            'Get tested if exposed',
            'Maintain good nutrition'
        ],
        'when_to_see_doctor': 'Seek immediate care if you cough up blood or have persistent cough with fever.',
        'severity': 'High',
        'confidence': 0.88
    },
    'covid19': {
        'keywords': ['covid', 'coronavirus', 'covid-19', 'loss of taste', 'loss of smell', 'dry cough', 'fever', 'shortness of breath', 'pandemic'],
        'description': 'COVID-19 is a contagious disease caused by the SARS-CoV-2 virus.',
        'symptoms': [
            'Fever or chills',
            'Cough',
            'Shortness of breath',
            'Fatigue',
            'Loss of taste or smell',
            'Sore throat',
            'Body aches',
            'Headache'
        ],
        'precautions': [
            'Get vaccinated and boosted',
            'Wear masks in crowded places',
            'Practice social distancing',
            'Wash hands frequently',
            'Isolate if positive',
            'Monitor oxygen levels',
            'Rest and stay hydrated'
        ],
        'when_to_see_doctor': 'Seek emergency care for difficulty breathing, chest pain, or bluish lips.',
        'severity': 'Moderate to High',
        'confidence': 0.89
    },
    'stroke': {
        'keywords': ['stroke', 'facial drooping', 'arm weakness', 'speech difficulty', 'sudden numbness', 'confusion', 'severe headache'],
        'description': 'A stroke occurs when blood supply to part of the brain is interrupted.',
        'symptoms': [
            'Sudden numbness or weakness (face, arm, leg)',
            'Confusion or trouble speaking',
            'Trouble seeing',
            'Difficulty walking',
            'Dizziness',
            'Severe headache',
            'Loss of balance'
        ],
        'precautions': [
            'Call emergency services immediately (FAST: Face, Arms, Speech, Time)',
            'Control blood pressure',
            'Manage diabetes',
            'Quit smoking',
            'Exercise regularly',
            'Eat healthy diet',
            'Limit alcohol'
        ],
        'when_to_see_doctor': 'CALL 911 IMMEDIATELY if you suspect a stroke. Time is critical!',
        'severity': 'Critical',
        'confidence': 0.92
    },
    'heart_attack': {
        'keywords': ['heart attack', 'chest pain', 'chest pressure', 'shortness of breath', 'arm pain', 'jaw pain', 'cardiac', 'myocardial infarction'],
        'description': 'A heart attack occurs when blood flow to the heart is blocked.',
        'symptoms': [
            'Chest pain or pressure',
            'Pain in arms, back, neck, jaw',
            'Shortness of breath',
            'Cold sweat',
            'Nausea',
            'Lightheadedness',
            'Fatigue'
        ],
        'precautions': [
            'Call emergency services immediately',
            'Chew aspirin if available',
            'Rest and stay calm',
            'Control risk factors (blood pressure, cholesterol)',
            'Exercise regularly',
            'Eat heart-healthy diet',
            'Quit smoking'
        ],
        'when_to_see_doctor': 'CALL 911 IMMEDIATELY if you suspect a heart attack!',
        'severity': 'Critical',
        'confidence': 0.93
    }
}

# Professional Medical UI Design
def load_advanced_css():
    """Load professional medical UI with excellent readability."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        animation: gradientShift 15s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main .block-container {
        max-width: 1400px;
        padding: 2rem 3rem;
    }
    
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    .app-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
        padding: 4rem 3rem;
        border-radius: 30px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.5);
        animation: floatHeader 3s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    @keyframes floatHeader {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .app-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: titlePulse 2s ease-in-out infinite;
        position: relative;
        z-index: 1;
    }
    
    @keyframes titlePulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .app-subtitle {
        font-size: 1.4rem;
        color: #4a5568 !important;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    .input-section {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 25px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        animation: slideUp 0.6s ease;
    }
    
    .input-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .input-section h2, .input-section h3 {
        color: #1a1a1a !important;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    
    .medical-card {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 25px;
        padding: 3rem;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
        border: 2px solid rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        animation: fadeIn 0.8s ease;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .medical-card h1, .medical-card h2, .medical-card h3, .medical-card h4 {
        color: #1a202c !important;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    
    .medical-card p, .medical-card span, .medical-card div, .medical-card li {
        color: #2d3748 !important;
        line-height: 2;
        font-size: 1.05rem;
    }
    
    .result-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        margin: 0.5rem;
        animation: popIn 0.5s ease;
    }
    
    @keyframes popIn {
        0% { transform: scale(0); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 1rem 3rem !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5) !important;
        width: 100% !important;
        position: relative;
        overflow: hidden;
        animation: buttonGlow 2s ease-in-out infinite;
    }
    
    @keyframes buttonGlow {
        0%, 100% {
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
        }
        50% {
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.8);
        }
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.7) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98) !important;
    }
    
    label {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stTextArea textarea, .stTextInput input, .stSelectbox select {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 15px !important;
        color: #1a202c !important;
        font-size: 1.05rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus, .stSelectbox select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2) !important;
        transform: translateY(-2px);
        background: #ffffff !important;
    }
    
    .stTextArea textarea:hover, .stTextInput input:hover, .stSelectbox select:hover {
        border-color: #a0aec0 !important;
    }
    
    .stSuccess {
        background: #d4edda !important;
        color: #155724 !important;
        border: 1px solid #c3e6cb !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stInfo {
        background: #d1ecf1 !important;
        color: #0c5460 !important;
        border: 1px solid #bee5eb !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stWarning {
        background: #fff3cd !important;
        color: #856404 !important;
        border: 1px solid #ffeaa7 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stError {
        background: #f8d7da !important;
        color: #721c24 !important;
        border: 1px solid #f5c6cb !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    p, span, div, li, td, th {
        color: #2d3748 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1a202c !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .app-title {
            font-size: 2.5rem;
        }
        .app-subtitle {
            font-size: 1rem;
        }
        .input-section, .medical-card {
            padding: 1.5rem;
        }
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Selection color */
    ::selection {
        background: #667eea;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def create_header():
    """Create application header."""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🏥 Advanced AI Medical Assistant</h1>
        <p class="app-subtitle">⚡ Instant Medical Analysis • 🎯 30+ Conditions • 💯 Built-in Knowledge Base</p>
    </div>
    """, unsafe_allow_html=True)

def analyze_symptoms(symptoms_text):
    """Analyze symptoms using built-in medical knowledge base."""
    symptoms_lower = symptoms_text.lower()
    
    # Score each condition
    condition_scores = {}
    for condition, data in MEDICAL_DATABASE.items():
        score = 0
        matched_keywords = []
        
        for keyword in data['keywords']:
            if keyword in symptoms_lower:
                score += 1
                matched_keywords.append(keyword)
                # Bonus for exact phrase match
                if f" {keyword} " in f" {symptoms_lower} ":
                    score += 0.5
        
        if score > 0:
            condition_scores[condition] = {
                'score': score,
                'data': data,
                'matched_keywords': matched_keywords
            }
    
    # Sort by score
    sorted_conditions = sorted(condition_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    if not sorted_conditions:
        return None
    
    # Return top match
    top_condition = sorted_conditions[0]
    condition_name = top_condition[0]
    condition_info = top_condition[1]['data']
    
    return {
        'condition': condition_name.replace('_', ' ').title(),
        'description': condition_info['description'],
        'symptoms': condition_info['symptoms'],
        'precautions': condition_info['precautions'],
        'when_to_see_doctor': condition_info['when_to_see_doctor'],
        'severity': condition_info['severity'],
        'confidence': min(0.95, condition_info['confidence'] + (top_condition[1]['score'] * 0.02)),
        'matched_keywords': top_condition[1]['matched_keywords']
    }

def main():
    """Main application function."""
    load_advanced_css()
    create_header()
    
    # Stats cards
    st.markdown("""
    <div style='display: flex; justify-content: space-around; flex-wrap: wrap; margin: 2rem 0; gap: 1rem;'>
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 20px; text-align: center; flex: 1; min-width: 200px;
                    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); animation: slideUp 0.6s ease;'>
            <h2 style='color: white; font-size: 2.5rem; margin: 0;'>30+</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>Medical Conditions</p>
        </div>
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 20px; text-align: center; flex: 1; min-width: 200px;
                    box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3); animation: slideUp 0.7s ease;'>
            <h2 style='color: white; font-size: 2.5rem; margin: 0;'>⚡</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>Instant Analysis</p>
        </div>
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 20px; text-align: center; flex: 1; min-width: 200px;
                    box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3); animation: slideUp 0.8s ease;'>
            <h2 style='color: white; font-size: 2.5rem; margin: 0;'>100%</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>No API Required</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Emergency section at top
    create_emergency_section()
    
    # Health tip
    create_health_tips()
    
    # Symptom checker
    selected_symptoms = create_symptom_checker()
    
    # Main input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h2>📝 Describe Your Symptoms</h2>', unsafe_allow_html=True)
    
    # Pre-fill if quick symptoms selected
    default_text = st.session_state.get('quick_symptoms', '')
    
    # Language selection
    language = st.selectbox(
        "Select Language:",
        ["English", "Hindi", "Spanish", "French", "German", "Italian"],
        help="Choose your preferred language for analysis"
    )
    
    # Symptoms input
    symptoms = st.text_area(
        "Describe your symptoms in detail:",
        value=default_text,
        height=150,
        placeholder="Please provide detailed information about your symptoms, including:\n• Duration of symptoms\n• Severity (1-10 scale)\n• Associated symptoms\n• Any triggers or patterns\n• Previous medical history related to these symptoms",
        help="Be as detailed as possible for better analysis"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Symptoms", 
            type="primary", 
            use_container_width=True,
            help="Click to start medical analysis"
        )
    
    # Process analysis
    if analyze_button:
        if not symptoms or not symptoms.strip():
            st.error("❌ Please enter your symptoms before analysis.")
            return
        
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown('<h2>🔍 Medical Analysis Results</h2>', unsafe_allow_html=True)
        
        # Enhanced loading animation
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        import time
        steps = [
            "🔍 Scanning symptoms...",
            "🧠 Analyzing patterns...",
            "📊 Calculating confidence...",
            "🏥 Matching conditions...",
            "✅ Generating results..."
        ]
        
        for i, step in enumerate(steps):
            progress_text.markdown(f"<h3 style='text-align: center; color: #667eea;'>{step}</h3>", unsafe_allow_html=True)
            progress_bar.progress((i + 1) / len(steps))
            time.sleep(0.3)
        
        result = analyze_symptoms(symptoms)
        progress_text.empty()
        progress_bar.empty()
        
        if not result:
            st.warning("⚠️ Could not identify a specific condition based on the symptoms provided. Please consult a healthcare provider for proper diagnosis.")
        else:
            # Display results with enhanced styling
            st.markdown(f"""
                <div style='text-align: center; margin: 2rem 0; padding: 2rem; 
                            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                            border-radius: 20px; border: 2px solid rgba(102, 126, 234, 0.3);'>
                    <h1 style='font-size: 3rem; color: #667eea; margin-bottom: 0.5rem; font-weight: 800;'>🏥 {result['condition']}</h1>
                    <p style='font-size: 1.3rem; color: #4a5568; font-style: italic; margin-top: 0.5rem;'>Comprehensive Medical Analysis & Recommendations</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Description with icon
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%); 
                            padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0;
                            border-left: 5px solid #667eea;'>
                    <h3 style='color: #667eea; margin-bottom: 1rem;'>📖 Description</h3>
                    <p style='font-size: 1.1rem; line-height: 1.8; color: #2d3748;'>{result['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Symptoms section
            st.markdown("""
                <div style='background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%); 
                            padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0;
                            border-left: 5px solid #3b82f6;'>
                    <h3 style='color: #3b82f6; margin-bottom: 1rem;'>🔍 Common Symptoms</h3>
            """, unsafe_allow_html=True)
            for symptom in result['symptoms']:
                st.markdown(f"<p style='margin: 0.5rem 0; font-size: 1.05rem;'>✓ {symptom}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Precautions section
            st.markdown("""
                <div style='background: linear-gradient(135deg, #d1fae5 0%, #d1f4e0 100%); 
                            padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0;
                            border-left: 5px solid #10b981;'>
                    <h3 style='color: #10b981; margin-bottom: 1rem;'>⚕️ Recommended Precautions</h3>
            """, unsafe_allow_html=True)
            for i, precaution in enumerate(result['precautions'], 1):
                st.markdown(f"<p style='margin: 0.5rem 0; font-size: 1.05rem;'>{i}. {precaution}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # When to see doctor section
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                            padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0;
                            border-left: 5px solid #ef4444;'>
                    <h3 style='color: #ef4444; margin-bottom: 1rem;'>🚨 When to See a Doctor</h3>
                    <p style='font-size: 1.1rem; line-height: 1.8; color: #7f1d1d; font-weight: 600;'>{result['when_to_see_doctor']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional Information Sections
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                                padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
                                border-left: 5px solid #f59e0b; height: 100%;'>
                        <h3 style='color: #f59e0b; margin-bottom: 1rem;'>💡 Prevention Tips</h3>
                        <p style='font-size: 1rem; line-height: 1.7; color: #78350f;'>
                            • Maintain a healthy lifestyle with regular exercise<br>
                            • Eat a balanced diet rich in fruits and vegetables<br>
                            • Get adequate sleep (7-9 hours nightly)<br>
                            • Stay hydrated throughout the day<br>
                            • Manage stress through relaxation techniques<br>
                            • Avoid smoking and limit alcohol consumption
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); 
                                padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
                                border-left: 5px solid #6366f1; height: 100%;'>
                        <h3 style='color: #6366f1; margin-bottom: 1rem;'>🏃 Lifestyle Recommendations</h3>
                        <p style='font-size: 1rem; line-height: 1.7; color: #312e81;'>
                            • Regular physical activity (30 min/day)<br>
                            • Maintain healthy weight (BMI 18.5-24.9)<br>
                            • Practice good hygiene habits<br>
                            • Schedule regular health check-ups<br>
                            • Keep track of your symptoms<br>
                            • Build a strong support system
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Did You Know section
            st.markdown("""
                <div style='background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); 
                            padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0;
                            border-left: 5px solid #ec4899;'>
                    <h3 style='color: #ec4899; margin-bottom: 1rem;'>💭 Did You Know?</h3>
                    <p style='font-size: 1.05rem; line-height: 1.8; color: #831843;'>
                        Early detection and proper management of health conditions can significantly improve outcomes. 
                        Regular health screenings, maintaining a healthy lifestyle, and staying informed about your health 
                        are key factors in preventing and managing medical conditions effectively.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Important Note
            st.markdown("""
                <div style='background: linear-gradient(135deg, #ddd6fe 0%, #c4b5fd 100%); 
                            padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0;
                            border: 2px solid #8b5cf6;'>
                    <h3 style='color: #7c3aed; margin-bottom: 1rem;'>📌 Important Note</h3>
                    <p style='font-size: 1rem; line-height: 1.7; color: #5b21b6; font-weight: 500;'>
                        This analysis is based on a comprehensive medical knowledge base and is intended for informational 
                        purposes only. It should not replace professional medical advice, diagnosis, or treatment. 
                        Always consult with qualified healthcare providers for accurate diagnosis and personalized treatment plans.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='text-align: center; margin: 2rem 0;'>
                <div style='display: inline-block; background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                            color: white; padding: 1rem 2rem; border-radius: 50px; font-weight: 600;
                            box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);'>
                    ✅ Analysis Completed Successfully!
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Export functionality
            export_text = export_results(result, symptoms)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=export_text,
                    file_name=f"medical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        # Add timestamp
        st.markdown(f"<p style='text-align: right; color: #718096; font-size: 0.9rem; margin-top: 2rem;'>Analysis generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def create_symptom_checker():
    """Interactive symptom checker with checkboxes."""
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h2>✅ Quick Symptom Checker</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #4a5568; margin-bottom: 1rem;">Select symptoms you are experiencing:</p>', unsafe_allow_html=True)
    
    common_symptoms = {
        'General': ['Fever', 'Fatigue', 'Weakness', 'Weight Loss', 'Weight Gain'],
        'Respiratory': ['Cough', 'Shortness of Breath', 'Wheezing', 'Chest Pain', 'Sore Throat'],
        'Digestive': ['Nausea', 'Vomiting', 'Diarrhea', 'Constipation', 'Abdominal Pain'],
        'Neurological': ['Headache', 'Dizziness', 'Confusion', 'Memory Loss', 'Numbness'],
        'Other': ['Joint Pain', 'Muscle Aches', 'Skin Rash', 'Itching', 'Swelling']
    }
    
    selected_symptoms = []
    cols = st.columns(3)
    
    for idx, (category, symptoms) in enumerate(common_symptoms.items()):
        with cols[idx % 3]:
            st.markdown(f"**{category}**")
            for symptom in symptoms:
                if st.checkbox(symptom, key=f"symptom_{symptom}"):
                    selected_symptoms.append(symptom)
    
    if selected_symptoms:
        st.success(f"✅ Selected {len(selected_symptoms)} symptom(s): {', '.join(selected_symptoms)}")
        symptom_text = f"I am experiencing: {', '.join(selected_symptoms)}"
        st.session_state['quick_symptoms'] = symptom_text
    
    st.markdown('</div>', unsafe_allow_html=True)
    return selected_symptoms

def create_emergency_section():
    """Emergency contacts and quick actions."""
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0;
                border: 3px solid #ef4444;'>
        <h2 style='color: #ef4444; margin-bottom: 1rem;'>🚨 Emergency Information</h2>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;'>
            <div style='background: white; padding: 1rem; border-radius: 10px;'>
                <h4 style='color: #ef4444; margin: 0;'>📞 Emergency</h4>
                <p style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0; color: #1a202c;'>911</p>
            </div>
            <div style='background: white; padding: 1rem; border-radius: 10px;'>
                <h4 style='color: #3b82f6; margin: 0;'>🏥 Poison Control</h4>
                <p style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0; color: #1a202c;'>1-800-222-1222</p>
            </div>
            <div style='background: white; padding: 1rem; border-radius: 10px;'>
                <h4 style='color: #10b981; margin: 0;'>💬 Crisis Hotline</h4>
                <p style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0; color: #1a202c;'>988</p>
            </div>
        </div>
        <p style='margin-top: 1rem; color: #7f1d1d; font-weight: 600;'>
            ⚠️ If you're experiencing a medical emergency, call 911 immediately!
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_health_tips():
    """Daily health tips carousel."""
    tips = [
        "💧 Drink at least 8 glasses of water daily for optimal hydration",
        "🏃 Exercise for 30 minutes daily to maintain cardiovascular health",
        "🥗 Eat 5 servings of fruits and vegetables every day",
        "😴 Get 7-9 hours of quality sleep each night",
        "🧘 Practice stress management through meditation or yoga",
        "🚭 Avoid smoking and limit alcohol consumption",
        "🦷 Maintain good oral hygiene by brushing twice daily",
        "☀️ Get 15 minutes of sunlight daily for Vitamin D",
        "🤝 Stay socially connected with friends and family",
        "📅 Schedule regular health check-ups and screenings"
    ]
    
    import random
    tip = random.choice(tips)
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0;
                border-left: 5px solid #3b82f6; text-align: center;'>
        <h3 style='color: #1e40af; margin-bottom: 1rem;'>💡 Daily Health Tip</h3>
        <p style='font-size: 1.2rem; color: #1e3a8a; font-weight: 500; margin: 0;'>{tip}</p>
    </div>
    """, unsafe_allow_html=True)

def export_results(result, symptoms):
    """Export analysis results as text."""
    if result:
        export_text = f"""
MEDICAL ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

CONDITION: {result['condition']}

SYMPTOMS ANALYZED:
{symptoms}

DESCRIPTION:
{result['description']}

COMMON SYMPTOMS:
{chr(10).join(['• ' + s for s in result['symptoms']])}

RECOMMENDED PRECAUTIONS:
{chr(10).join([f'{i+1}. {p}' for i, p in enumerate(result['precautions'])])}

WHEN TO SEE A DOCTOR:
{result['when_to_see_doctor']}

{'='*50}
DISCLAIMER: This analysis is for informational purposes only.
Always consult with qualified healthcare providers.
"""
        return export_text
    return ""

if __name__ == "__main__":
    main()
