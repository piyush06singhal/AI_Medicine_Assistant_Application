# 🔑 API Integration Guide

## How to Add Real AI to Your Medical App

Currently, your app uses a built-in medical knowledge base. Here's how to integrate real AI services:

---

## Option 1: OpenAI GPT-4 Integration

### Step 1: Install OpenAI Package

```bash
pip install openai
```

### Step 2: Add to requirements.txt

```
openai>=1.0.0
```

### Step 3: Modify the Code

Replace the `enhanced_disease_prediction()` function with:

```python
import openai

def enhanced_disease_prediction(symptoms, language='en', include_risk_factors=True, detailed_analysis=True):
    """Enhanced disease prediction using OpenAI GPT-4."""

    # Check if API key is available
    api_key = st.session_state.get('api_key', None)

    if api_key:
        # Use OpenAI API
        openai.api_key = api_key

        prompt = f"""You are a medical AI assistant. Analyze these symptoms and provide:
        1. Most likely condition
        2. Confidence level (0-1)
        3. Related symptoms
        4. Precautions
        5. Risk factors

        Symptoms: {symptoms}
        Language: {language}

        Respond in JSON format."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            # Parse AI response
            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            st.error(f"API Error: {str(e)}")
            # Fall back to knowledge base
            return use_knowledge_base(symptoms, language, include_risk_factors, detailed_analysis)
    else:
        # Use built-in knowledge base
        return use_knowledge_base(symptoms, language, include_risk_factors, detailed_analysis)
```

---

## Option 2: Google Gemini Integration

### Step 1: Install Google AI Package

```bash
pip install google-generativeai
```

### Step 2: Add to requirements.txt

```
google-generativeai>=0.3.0
```

### Step 3: Modify the Code

```python
import google.generativeai as genai

def enhanced_disease_prediction(symptoms, language='en', include_risk_factors=True, detailed_analysis=True):
    """Enhanced disease prediction using Google Gemini."""

    api_key = st.session_state.get('api_key', None)

    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')

        prompt = f"""Analyze these medical symptoms and provide detailed analysis:

        Symptoms: {symptoms}

        Provide:
        1. Most likely condition
        2. Confidence score (0-100%)
        3. Related symptoms (as paragraph)
        4. Recommended actions (as paragraph)
        5. Risk factors (as paragraph)

        Format as natural paragraphs, not bullet points."""

        try:
            response = model.generate_content(prompt)
            # Parse and format response
            return parse_gemini_response(response.text)

        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return use_knowledge_base(symptoms, language, include_risk_factors, detailed_analysis)
    else:
        return use_knowledge_base(symptoms, language, include_risk_factors, detailed_analysis)
```

---

## Option 3: Hugging Face Medical Models

### Step 1: Install Transformers

```bash
pip install transformers torch
```

### Step 2: Use Medical-Specific Models

```python
from transformers import pipeline

# Load medical NLP model
medical_classifier = pipeline("text-classification", model="medical-ner-model")

def enhanced_disease_prediction(symptoms, language='en', include_risk_factors=True, detailed_analysis=True):
    """Enhanced disease prediction using Hugging Face models."""

    try:
        # Use medical NLP model
        predictions = medical_classifier(symptoms)

        # Process predictions
        result = {
            'predicted_disease': predictions[0]['label'],
            'confidence': predictions[0]['score'],
            # ... format other fields
        }

        return result

    except Exception as e:
        st.error(f"Model Error: {str(e)}")
        return use_knowledge_base(symptoms, language, include_risk_factors, detailed_analysis)
```

---

## Getting API Keys

### OpenAI

1. Go to https://platform.openai.com/
2. Sign up / Log in
3. Navigate to API Keys
4. Create new secret key
5. Copy and save securely

**Cost**: ~$0.03 per 1K tokens (GPT-4)

### Google Gemini

1. Go to https://makersuite.google.com/
2. Sign in with Google account
3. Get API key
4. Copy and save securely

**Cost**: Free tier available, then pay-as-you-go

### Hugging Face

1. Go to https://huggingface.co/
2. Sign up / Log in
3. Go to Settings → Access Tokens
4. Create new token
5. Copy and save securely

**Cost**: Free for most models

---

## Security Best Practices

### 1. Never Hardcode API Keys

```python
# ❌ BAD
api_key = "sk-1234567890abcdef"

# ✅ GOOD
api_key = st.session_state.get('api_key', None)
```

### 2. Use Environment Variables

```python
import os
api_key = os.getenv('OPENAI_API_KEY')
```

### 3. Add to .gitignore

```
.env
*.key
secrets.toml
```

### 4. Use Streamlit Secrets (for deployment)

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-key-here"
GEMINI_API_KEY = "your-key-here"
```

Access in code:

```python
api_key = st.secrets.get("OPENAI_API_KEY", None)
```

---

## Testing Your Integration

### 1. Test with Sample Symptoms

```python
test_symptoms = "I have a headache, fever, and body aches for 3 days"
result = enhanced_disease_prediction(test_symptoms)
print(result)
```

### 2. Check Error Handling

```python
# Test with invalid API key
st.session_state.api_key = "invalid-key"
result = enhanced_disease_prediction(test_symptoms)
# Should fall back to knowledge base
```

### 3. Monitor API Usage

- Check your API dashboard regularly
- Set usage limits
- Monitor costs

---

## Recommended Approach

**For Production:**

1. Start with built-in knowledge base (current system)
2. Add OpenAI GPT-4 for premium users
3. Use Gemini as fallback (free tier)
4. Keep knowledge base as final fallback

**For Development:**

1. Test with Gemini (free tier)
2. Validate results
3. Switch to GPT-4 for production
4. Monitor performance and costs

---

## Cost Estimation

### OpenAI GPT-4

- Average query: ~500 tokens
- Cost per query: ~$0.015
- 1000 queries: ~$15

### Google Gemini

- Free tier: 60 queries/minute
- Paid: ~$0.001 per query
- 1000 queries: ~$1

### Hugging Face

- Free for inference
- Can host your own models
- No per-query costs

---

## Need Help?

If you want to implement any of these integrations, let me know which service you prefer and I'll help you set it up!

**Current Status**: Your app works perfectly with the built-in knowledge base. API integration is optional for enhanced capabilities.
