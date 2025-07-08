import streamlit as st
from langchain.llms import HuggingFaceHub  
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
st.set_page_config(page_title="AI Flashcard Generator", layout="wide")
st.title("📚 AI Flashcard Generator")
st.caption("Paste your notes or textbook content to auto-generate flashcards using Hugging Face!")

# Use token from Streamlit secrets (fallback to manual input)
hf_token = st.secrets.get("HUGGINGFACE_API_TOKEN", None)
if not hf_token:
    hf_token = st.text_input("🔑 Enter your Hugging Face API Token", type="password")

text = st.text_area("📄 Paste your notes here", height=300)

if st.button("⚡ Generate Flashcards") and hf_token and text:
    with st.spinner("Thinking..."):
        try:
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",
                huggingfacehub_api_token=hf_token,
                model_kwargs={"temperature": 0.5, "max_length": 512}
            )

            prompt = PromptTemplate(
                input_variables=["content"],
                template="""
You are a helpful assistant that creates educational flashcards.

Given the following study content, generate a list of flashcards in the format:
Question: ...
Answer: ...

Content:
\"\"\"{content}\"\"\"

Flashcards:
"""
            )

            chain = LLMChain(llm=llm, prompt=prompt)
            output = chain.run(content=text)

            flashcards = []
            for block in output.split("Question:")[1:]:
                if "Answer:" in block:
                    q, a = block.split("Answer:", 1)
                    flashcards.append({"Question": q.strip(), "Answer": a.strip()})

            df = pd.DataFrame(flashcards)
            st.success(f"✅ Generated {len(df)} flashcards!")

            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", csv, "flashcards.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")
