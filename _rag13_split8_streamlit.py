
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import io
from PIL import Image
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# ✅ OpenAI APIキー
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が設定されていません。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ 分割された embeddings ファイルをすべて結合
embeddings = []
dfs = []
for part_num in range(1, 9):
    with open(f"embeddings_part{part_num}.pkl", "rb") as f:
        part_data = pickle.load(f)
        embeddings.extend(part_data["embeddings"])
        dfs.append(part_data["df"])

embeddings = np.array(embeddings)
df = pd.concat(dfs, ignore_index=True)

# ✅ 類似問題検索関数
def find_similar_questions(query_text, top_k=5):
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    ).data[0].embedding

    query_vec = np.array(query_embedding).reshape(1, -1)
    sim_scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = sim_scores.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]

# ✅ Streamlit UI
st.title("📷 歯科医師国家試験・画像問題AI解析")

uploaded_file = st.file_uploader("国家試験問題の画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner("画像をGPT-4oで解析中..."):
        image = Image.open(uploaded_file)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        base64_image = base64.b64encode(image_bytes.read()).decode()

        vision_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "あなたは国家試験のOCR専門家です。画像から問題文と選択肢（a〜e）を正確に抽出してください。"
                },
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]
                }
            ],
            max_tokens=1000
        )

        extracted_question = vision_response.choices[0].message.content.strip()

    with st.spinner("類似問題を検索中..."):
        similar_df = find_similar_questions(extracted_question, top_k=5)

    with st.spinner("解説と類題を生成中（GPT-4o）..."):
        similar_texts = "\n\n".join(
            f"{i+1}. {row['設問']}\na. {row['選択肢a']}\nb. {row['選択肢b']}\nc. {row['選択肢c']}\nd. {row['選択肢d']}\ne. {row['選択肢e']}\n正解: {row['正解']}"
            for i, (_, row) in enumerate(similar_df.iterrows())
        )

        final_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "system",
                    "content": "あなたは国家試験問題の教育専門家です。以下の画像から抽出した問題と、それに類似する過去問を参考にして、次の内容を生成してください：\n1. 出題の意図\n2. 正解\n3. 各選択肢の解説\n4. 類題3問（それぞれ正解と解説付き）"
                },
                {
                    "role": "user",
                    "content": f"【画像問題】\n{extracted_question}\n\n【類似問題】\n{similar_texts}"
                }
            ],
            max_tokens=2000
        )

        st.markdown("## 🧠 GPT-4oによる解析結果")
        st.markdown(final_response.choices[0].message.content.strip())
