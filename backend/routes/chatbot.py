from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import datetime
from backend.database import get_db
from backend.models import User, Customer, Product, Purchase, Complaint
from backend.auth import get_current_user
from backend.ml.recommendation import SVDRecommender
import pickle
import os

router = APIRouter(prefix="/api/chatbot", tags=["AI Chatbot"])

class ChatRequest(BaseModel):
    message: str
    customer_id: Optional[int] = None

# FAQ Data
FAQ_RESPONSES = {
    "shipping": "We offer standard shipping (3-5 business days) and express shipping (1-2 business days). Standard shipping is free for orders over $50.",
    "return": "You can return any genuine, unused product in its original packaging within 30 days of purchase. Simply visit your Profile, select Purchase History, and click 'Initiate Return'.",
    "refund": "Once we receive and inspect your returned product, refunds are processed within 5-7 business days to your original payment method.",
    "fake": "To detect a counterfeit product, go to the 'Counterfeit Scan' tab on the dashboard. Upload the product packaging image and company logo. Our AI will analyze differences and highlight mismatches.",
    "support": "You can reach our customer support team via email at support@intelligentcrm.com or register a complaint directly through this chat by saying 'register a complaint'."
}

@router.post("/message")
def chat_with_bot(request: ChatRequest, db: Session = Depends(get_db)):
    user_msg = request.message.lower().strip()
    cust_id = request.customer_id
    
    # 1. Identity Check
    customer = None
    if cust_id:
        customer = db.query(Customer).filter(Customer.id == cust_id).first()

    # 2. Handle Complaint Registration
    if "complaint" in user_msg or "register a complaint" in user_msg or "file a complaint" in user_msg:
        if not customer:
            return {
                "response": "To file a complaint, you must be logged in. Please log in first.",
                "context": "auth_required"
            }
        
        # Extract complaint text from message or prompt them
        # Let's check if they provided details
        parts = request.message.split(":", 1)
        if len(parts) > 1:
            complaint_text = parts[1].strip()
        else:
            complaint_text = request.message
            
        # Log complaint
        new_complaint = Complaint(
            customer_id=customer.id,
            complaint_text=f"Logged via Chatbot: {complaint_text}",
            status="Open"
        )
        db.add(new_complaint)
        db.commit()
        
        return {
            "response": f"Thank you, {customer.name}. Your complaint has been registered successfully with status 'Open'. Our support team will resolve this shortly.",
            "context": "complaint_registered"
        }

    # 3. Handle Order Tracking
    if "track" in user_msg or "order" in user_msg or "history" in user_msg:
        if not customer:
            return {
                "response": "I can help you track your orders! Please log in to view your purchase history.",
                "context": "auth_required"
            }
        purchases = db.query(Purchase).filter(Purchase.customer_id == customer.id).order_by(Purchase.purchase_date.desc()).all()
        if not purchases:
            return {
                "response": f"Hi {customer.name}, you haven't made any purchases yet. Browse our products catalog to place your first order!",
                "context": "order_tracking"
            }
            
        # Format recent order
        recent = purchases[0]
        prod = db.query(Product).filter(Product.id == recent.product_id).first()
        prod_name = prod.name if prod else "Product"
        
        response_text = f"Hi {customer.name}, your most recent order was for **{recent.quantity}x {prod_name}** on {recent.purchase_date.strftime('%Y-%m-%d')}. Status: **In Transit** (Estimated Delivery: in 2 days)."
        if len(purchases) > 1:
            response_text += f"\n\nYou have {len(purchases)} items in your purchase history. Click 'Purchase History' to see the full list."
            
        return {
            "response": response_text,
            "context": "order_tracking"
        }

    # 4. Handle Recommendations request
    if "recommend" in user_msg or "suggest" in user_msg or "personalized" in user_msg:
        if not customer:
            # Cold-start trending recommendations
            products = db.query(Product).limit(3).all()
            prod_names = ", ".join([p.name for p in products])
            return {
                "response": f"Log in for personalized recommendations! Meanwhile, here are some trending products on our platform: {prod_names}.",
                "context": "general_recommendations"
            }
            
        # Load SVD recommender
        rec_path = "backend/ml/models/recommender.pkl"
        if os.path.exists(rec_path):
            with open(rec_path, "rb") as f:
                recommender = pickle.load(f)
            rec_ids = recommender.recommend_personalized(customer.id, top_n=3)
            products = db.query(Product).filter(Product.id.in_(rec_ids)).all()
        else:
            products = db.query(Product).limit(3).all()
            
        prod_list_str = "\n".join([f"- **{p.name}** (${p.price:.2f}) by {p.brand_name}" for p in products])
        return {
            "response": f"Hi {customer.name}, based on your interests, here are my top product recommendations for you:\n\n{prod_list_str}",
            "context": "personalized_recommendations"
        }

    # 5. Handle Product Search
    if "search" in user_msg or "find" in user_msg or "buy" in user_msg:
        # Extract search keyword
        words = user_msg.split()
        keyword = None
        # Try to find words after "search" or "find"
        for i, word in enumerate(words):
            if word in ("search", "find", "buy") and i + 1 < len(words):
                keyword = " ".join(words[i+1:])
                break
        if not keyword:
            keyword = user_msg
            
        # Search DB
        matched = db.query(Product).filter(
            (Product.name.like(f"%{keyword}%")) | 
            (Product.category.like(f"%{keyword}%")) | 
            (Product.brand_name.like(f"%{keyword}%"))
        ).limit(3).all()
        
        if matched:
            result_str = "\n".join([f"- **{p.name}** (${p.price:.2f}) - {p.category}" for p in matched])
            return {
                "response": f"Here are the top products matching '{keyword}':\n\n{result_str}",
                "context": "product_search"
            }
        else:
            return {
                "response": f"Sorry, I couldn't find any products matching '{keyword}'. Try checking for other terms like 'iPhone', 'Nike', or 'Headphones'.",
                "context": "product_search"
            }

    # 6. Check FAQ answers
    for key, ans in FAQ_RESPONSES.items():
        if key in user_msg:
            return {
                "response": ans,
                "context": "faq"
            }

    # 7. Default greetings/fallback
    if "hello" in user_msg or "hi" in user_msg or "hey" in user_msg:
        greeting = f"Hello {customer.name}!" if customer else "Hello!"
        return {
            "response": f"{greeting} Welcome to our AI Assistant. I can help you search products, track orders, recommend personalized items, answer FAQs (shipping/returns), or file complaints.\n\nWhat can I assist you with today?",
            "context": "greeting"
        }

    return {
        "response": "I'm not sure I fully understand. You can ask me to:\n1. 'Search iPhone' (or another product)\n2. 'Track my order'\n3. 'Recommend products'\n4. Ask about 'shipping' or 'return' policy\n5. 'Register a complaint' followed by details",
        "context": "fallback"
    }
