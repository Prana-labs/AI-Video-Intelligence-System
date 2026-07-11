import os
import streamlit as st

def setup_groq_api():
    """Setup Groq API key"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 Groq API Setup")
    
    # Check if API key is already set
    api_key = os.getenv('GROQ_API_KEY')
    
    if api_key:
        st.sidebar.success("✅ GROQ_API_KEY is set!")
        st.sidebar.info(f"Key: {api_key[:10]}...")
    else:
        st.sidebar.warning("⚠️ GROQ_API_KEY not found!")
        st.sidebar.markdown("""
        To use Groq for better answers:

        1. Get free API key from [groq.com](https://console.groq.com)
        2. Set environment variable:
           ```bash
           set GROQ_API_KEY=your_api_key_here
           ```
        3. Restart the app

        **Free tier:** 100 requests/minute
        **Models available:** llama3.1-8b-instant, mixtral-8x7b-32768, etc.
        """)

if __name__ == "__main__":
    setup_groq_api()
