import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=groq_key
)

def rerank_and_explain_with_llm(query, movies_df, predicted_genres):
    # Drop duplicates on Title (keep first occurrence)
    deduped_df = movies_df.drop_duplicates(subset="Title").reset_index(drop=True)

    movie_list = ''
    for idx, row in deduped_df.iterrows():
        title = row.get("Title")
        overview = row.get("Overview")
        movie_list += f"{idx+1}. {title} - {overview}\n"

    # PROMPT
    prompts = [
        {
            "role": "system",
            "content": f"""
                            You are an intelligent movie assistant. A user gave a movie-related query. A list of movies and their overviews is provided. Based on the user's query and the predicted genres: {', '.join(predicted_genres)}, rank the movies from most to least relevant.

                            Your job:
                            1. Rank the movies below from most to least relevant to the user's query.
                            2. For each movie, explain briefly (1–2 sentences) why it fits the query.
                               - Be conversational and friendly.
                               - Do NOT invent movie titles or change them.
                               - Only use the titles exactly as given.

                            Return in this format:
                            1. <Movie Title> - <1–2 sentence explanation>
                            2. ...
                            Do NOT add commentary or extra info before or after the list.
                            DO NOT ALTER ANY INFORMATION , DO NOT ADD NEW INFORMATION.
                            DO NOT MAKE UP TITLES , DO NOT CHANGE TITLES.
                            STICK TO THE TITLES YOU HAVE BEEN GIVEN AND RETURN THOSE ONLY
                """
        },
        {
            "role": "user",
            "content": f'''
                        User's query: {query}

                        Movie list:
                        {movie_list}
                '''
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=prompts,
        model="llama-3.3-70b-versatile"
    )

    response = chat_completion.choices[0].message.content.strip()

    # Parse response
    ranked_titles = []
    explanations = []

    for line in response.split("\n"):
        match = re.match(r"^\d+\.\s*(.*?)\s*-\s*(.+)$", line.strip())
        if match:
            title = match.group(1).strip()
            explanation = match.group(2).strip()
            if title not in ranked_titles:
                ranked_titles.append(title)
                explanations.append(explanation)

    # Map titles back to original df (deduped)
    available_titles = set(deduped_df["Title"])
    filtered_titles, filtered_explanations = [], []

    for t, e in zip(ranked_titles, explanations):
        if t in available_titles:
            filtered_titles.append(t)
            filtered_explanations.append(e)

    # Handle missing titles (if LLM skipped some)
    missing_titles = [t for t in deduped_df["Title"] if t not in filtered_titles]
    for t in missing_titles:
        filtered_titles.append(t)
        filtered_explanations.append("Relevant aspects not clearly identified, but could be worth exploring.")

    # Final DataFrame
    reranked_df = deduped_df.set_index("Title").loc[filtered_titles].reset_index()
    reranked_df["Explanation"] = filtered_explanations

    return reranked_df
