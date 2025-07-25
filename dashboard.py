import gradio as gr
from main import retrieve_semantic_recommendations
import pandas as pd


def fetch_recommendations(query, num_results=5):
    """
    Fetch movie recommendations based on semantic query

    Args:
        query (str): User's movie preference description
        num_results (int): Number of recommendations to return

    Returns:
        pd.DataFrame: Formatted recommendations with HTML for images
    """
    try:
        # Get recommendations from main function
        movies_df = retrieve_semantic_recommendations(query)

        # Limit results based on user preference
        movies_df = movies_df.head(num_results)

        # Create enhanced display with HTML formatting for posters
        display_data = []
        for _, row in movies_df.iterrows():
            poster_html = f'<img src="{row["Poster_Url"]}" width="100" height="150" style="border-radius: 8px;">' if pd.notna(
                row.get("Poster_Url")) else "No Image"

            display_data.append({
                "🎬 Title": row["Title"],
                "📝 Explanation": row["Explanation"],
                "🖼️ Poster": poster_html
            })

        display_df = pd.DataFrame(display_data)
        return display_df

    except Exception as e:
        # Return error message as dataframe
        error_df = pd.DataFrame({
            "Error": [f"Failed to fetch recommendations: {str(e)}"],
            "Suggestion": ["Please try rephrasing your query or check your connection."]
        })
        return error_df


def get_example_queries():
    """Return list of example queries for user inspiration"""
    return [
        "I want a war movie where a soldier comes home to a missing family",
        "Romantic comedy with witty dialogue and charming leads",
        "Dark psychological thriller with unreliable narrator",
        "Sci-fi movie about artificial intelligence and humanity",
        "Coming-of-age story set in a small town",
        "Action movie with elaborate heist sequences",
        "Horror film with supernatural elements and jump scares"
    ]


# Create the interface with enhanced features
with gr.Blocks(
        title="🎬 Semantic Movie Recommender",
        theme=gr.themes.Soft(),
        css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .dataframe img {
        object-fit: cover;
    }
    """
) as iface:
    gr.Markdown("""
    # 🎬 Semantic Movie Recommender

    Describe the kind of movie you're in the mood for, and get personalized recommendations with explanations!

    **Examples:** Try describing a mood, genre, plot elements, or specific themes you want to explore.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="🔍 Describe your movie preferences",
                placeholder="e.g., 'I want something uplifting about overcoming challenges'",
                lines=3
            )

            num_results = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="📊 Number of recommendations"
            )

            submit_btn = gr.Button("🎭 Get Recommendations", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 💡 Example Queries")
            example_buttons = []
            for example in get_example_queries()[:4]:  # Show first 4 examples
                btn = gr.Button(example, size="sm")
                example_buttons.append(btn)
                btn.click(lambda x=example: x, outputs=query_input)

    # Results section
    results_df = gr.Dataframe(
        label="🎯 Your Movie Recommendations",
        datatype=["str", "str", "html"],  # Allow HTML in poster column
        wrap=True
    )

    # Event handlers
    submit_btn.click(
        fn=fetch_recommendations,
        inputs=[query_input, num_results],
        outputs=results_df
    )

    # Also trigger on Enter key
    query_input.submit(
        fn=fetch_recommendations,
        inputs=[query_input, num_results],
        outputs=results_df
    )

    # Footer with tips
    gr.Markdown("""
    ---
    ### 🎯 Tips for better recommendations:
    - Be specific about mood, themes, or plot elements
    - Mention preferred genres or time periods
    - Describe the emotional experience you're seeking
    - Include any actors, directors, or similar movies you enjoyed
    """)

if __name__ == "__main__":
    iface.launch(
        server_name="127.0.0.1",  # Allow external access
        server_port=7860,
        share=False,  # Set to True if you want a public link
        debug=True
    )