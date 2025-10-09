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
    }
}

# Professional Medical UI Design
def load_advanced_css():
    """Load professional medical UI with excellent readability."""
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 50%, #cbd5e0 100%);
    }
    
    .main .block-container {
        max-width: 1200px;
        padding: 2rem 3rem;
    }
    
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
    }
    
    .app-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .app-subtitle {
        font-size: 1.3rem;
        color: #f0f0f0 !important;
        font-weight: 400;
    }
    
    .input-section {
        background: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        border: 2px solid #e0e0e0;
    }
    
    .input-section h2, .input-section h3 {
        color: #1a1a1a !important;
        margin-bottom: 1rem;
    }
    
    .medical-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        border: 2px solid #e2e8f0;
    }
    
    .medical-card h1, .medical-card h2, .medical-card h3, .medical-card h4 {
        color: #1a202c !important;
        margin-bottom: 1rem;
    }
    
    .medical-card p, .medical-card span, .medical-card div, .medical-card li {
        color: #2d3748 !important;
        line-height: 1.8;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2.5rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.6) !important;
    }
    
    label {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stTextArea textarea, .stTextInput input, .stSelectbox select {
        background: #ffffff !important;
        border: 2px solid #cbd5e0 !important;
        border-radius: 10px !important;
        color: #1a202c !important;
        font-size: 1rem !important;
        padding: 0.75rem !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus, .stSelectbox select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
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
    </style>
    """, unsafe_allow_html=True)

def create_header():
    """Create application header."""
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🏥 Advanced AI Medical Assistant</h1>
        <p class="app-subtitle">Built-in medical knowledge base - No API required</p>
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
    
    # Main input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h2>📝 Describe Your Symptoms</h2>', unsafe_allow_html=True)
    
    # Language selection
    language = st.selectbox(
        "Select Language:",
        ["English", "Hindi", "Spanish", "French", "German", "Italian"],
        help="Choose your preferred language for analysis"
    )
    
    # Symptoms input
    symptoms = st.text_area(
        "Describe your symptoms in detail:",
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
    
    # Show disclaimer
    st.warning("⚠️ **Medical Disclaimer**: This tool provides information for educational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.")
    
    # Process analysis
    if analyze_button:
        if not symptoms or not symptoms.strip():
            st.error("❌ Please enter your symptoms before analysis.")
            return
        
        st.markdown('<div class="medical-card">', unsafe_allow_html=True)
        st.markdown('<h2>🔍 Medical Analysis Results</h2>', unsafe_allow_html=True)
        
        with st.spinner("🤖 Analyzing your symptoms..."):
            result = analyze_symptoms(symptoms)
            
            if not result:
                st.warning("⚠️ Could not identify a specific condition based on the symptoms provided. Please consult a healthcare provider for proper diagnosis.")
            else:
                # Display results
                st.markdown(f"### 🏥 Possible Condition: {result['condition']}")
                st.markdown(f"**Confidence Level:** {result['confidence']*100:.1f}%")
                st.markdown(f"**Severity:** {result['severity']}")
                
                st.markdown("---")
                
                st.markdown("### 📖 Description")
                st.markdown(result['description'])
                
                st.markdown("### 🔍 Common Symptoms")
                for symptom in result['symptoms']:
                    st.markdown(f"• {symptom}")
                
                st.markdown("### ⚕️ Recommended Precautions")
                for precaution in result['precautions']:
                    st.markdown(f"• {precaution}")
                
                st.markdown("### 🚨 When to See a Doctor")
                st.error(result['when_to_see_doctor'])
                
                st.success("✅ Analysis completed successfully!")
        
        # Add timestamp
        st.markdown(f"<p style='text-align: right; color: #718096; font-size: 0.9rem; margin-top: 2rem;'>Analysis generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
