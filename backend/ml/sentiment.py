import logging

logger = logging.getLogger(__name__)

# Try to import transformers and set up BERT pipeline
BERT_PIPELINE = None
try:
    from transformers import pipeline
    # Load a lightweight DistilBERT sentiment model
    BERT_PIPELINE = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # Force CPU to avoid CUDA setup issues
    )
    logger.info("BERT sentiment pipeline initialized successfully.")
except Exception as e:
    logger.warning(f"Failed to initialize HuggingFace BERT pipeline: {e}. Falling back to TextBlob.")

# Import TextBlob for robust offline fallback
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None


def analyze_sentiment(text: str) -> tuple[str, float]:
    """
    Analyzes sentiment of the text.
    Returns:
        sentiment: 'Positive', 'Neutral', or 'Negative'
        score: confidence score / polarity (between 0.0 and 1.0)
    """
    if not text or not text.strip():
        return "Neutral", 0.5

    # 1. Try BERT Sentiment Analysis
    if BERT_PIPELINE is not None:
        try:
            result = BERT_PIPELINE(text)[0]
            label = result["label"]  # 'POSITIVE' or 'NEGATIVE'
            score = float(result["score"])
            
            # Map labels
            if label == "POSITIVE":
                sentiment = "Positive"
                # If score is close to 0.5, we can call it Neutral
                if score < 0.6:
                    sentiment = "Neutral"
            else:
                sentiment = "Negative"
                if score < 0.6:
                    sentiment = "Neutral"
                    
            return sentiment, score
        except Exception as e:
            logger.warning(f"BERT analysis failed: {e}. Falling back to TextBlob.")

    # 2. Fallback to TextBlob
    if TextBlob is not None:
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # Range: [-1.0, 1.0]
            
            if polarity > 0.15:
                sentiment = "Positive"
            elif polarity < -0.15:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
                
            # Convert polarity to a pseudo-confidence score [0, 1]
            score = float((polarity + 1.0) / 2.0)
            return sentiment, score
        except Exception:
            pass

    # 3. Simple Lexicon Fallback (Zero dependencies)
    positive_words = {"good", "great", "excellent", "love", "awesome", "amazing", "happy", "satisfied", "best", "perfect", "nice"}
    negative_words = {"bad", "poor", "worst", "hate", "terrible", "fake", "broken", "angry", "disappointed", "slow", "defect"}
    
    words = text.lower().split()
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    
    if pos_count > neg_count:
        return "Positive", 0.8
    elif neg_count > pos_count:
        return "Negative", 0.8
    else:
        return "Neutral", 0.5
