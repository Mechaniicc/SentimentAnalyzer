import sys
import os

# Add parent directory to path to import from model directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from model.model import SentimentAnalyzer
import time
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis App", page_icon="🔍", layout="centered"
)


# Initialize sentiment analyzer with fine-tuned model
@st.cache_resource
def load_analyzer():
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model",
        "fine_tuned_model",
    )
    return SentimentAnalyzer(
        model_path=model_path if os.path.exists(model_path) else None
    )


def create_confidence_graph(confidence_scores):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 3))

    # Data for the bar chart
    sentiments = list(confidence_scores.keys())
    scores = list(confidence_scores.values())

    # Create horizontal bar chart with custom colors
    bars = ax.barh(
        sentiments,
        scores,
        color=["#ff9999", "#99ff99"],  # Red for negative, green for positive
    )

    # Customize the graph
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence Score (%)")
    ax.set_title("Sentiment Confidence Scores")

    # Add value labels on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            ha="left",
            va="center",
            fontweight="bold",
        )

    # Adjust layout
    plt.tight_layout()
    return fig


def main():
    st.title("📝 Review Sentiment Analyzer")
    st.write(
        "Analyze the sentiment of reviews using a DistilBERT model fine-tuned on IMDB dataset"
    )

    # Initialize sentiment analyzer
    analyzer = load_analyzer()

    # Create a text input area
    review_text = st.text_area(
        "Enter review text here", height=150, placeholder="Enter your review here..."
    )

    # Add a submit button
    if st.button("Analyze Sentiment"):
        if review_text:
            with st.spinner("Analyzing Text..."):
                # Add a small delay to simulate processing
                time.sleep(0.5)

                # Get sentiment analysis
                result = analyzer.analyze_sentiment(review_text)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    sentiment_color = (
                        "#99ff99" if result["sentiment"] == "positive" else "#ff9999"
                    )
                    st.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: {sentiment_color};">
                            <h3 style="margin: 0; color: black;">Sentiment: {result["sentiment"].capitalize()}</h3>
                            <p style="margin: 5px 0 0 0; color: black;">Confidence: {result["confidence"]:.1f}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.write("**Detailed Scores**")
                    for label, score in result["confidence_scores"].items():
                        st.write(f"{label.capitalize()}: {score:.1f}%")

                # Add confidence score visualization
                st.write("### Confidence Score Visualization")
                fig = create_confidence_graph(result["confidence_scores"])
                st.pyplot(fig)

                # Add a visual indicator of sentiment
                if result["sentiment"] == "positive":
                    st.success("This review expresses a positive sentiment! 👍")
                else:
                    st.error("This review expresses a negative sentiment. 👎")

        else:
            st.warning("Please enter some text to analyze.")


if __name__ == "__main__":
    main()
