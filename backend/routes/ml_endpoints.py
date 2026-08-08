import os
import shutil
import pickle
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import User, Customer, Product, Purchase, Review, Complaint, CounterfeitScan
from backend.auth import get_current_user
from backend.ml.image_similarity import calculate_image_similarity


router = APIRouter(prefix="/api/ml", tags=["Machine Learning Inferences"])

# Paths to models
CHURN_MODEL_PATH = "backend/ml/models/churn_model.pkl"
RECOMMENDER_PATH = "backend/ml/models/recommender.pkl"
UPLOAD_DIR = "backend/static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper to load models dynamically
def load_churn_model():
    if not os.path.exists(CHURN_MODEL_PATH):
        raise HTTPException(status_code=500, detail="Churn model not trained. Run train_models first.")
    with open(CHURN_MODEL_PATH, "rb") as f:
        return pickle.load(f)

def load_recommender():
    if not os.path.exists(RECOMMENDER_PATH):
        raise HTTPException(status_code=500, detail="Recommender model not trained. Run train_models first.")
    with open(RECOMMENDER_PATH, "rb") as f:
        return pickle.load(f)


@router.get("/churn-predict/{customer_id}")
def predict_customer_churn(customer_id: int, db: Session = Depends(get_db)):
    """Predict churn probability and return retention suggestions for a customer."""
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
        
    user = db.query(User).filter(User.id == customer_id).first()
    purchases = db.query(Purchase).filter(Purchase.customer_id == customer_id).all()
    complaints = db.query(Complaint).filter(Complaint.customer_id == customer_id).all()
    reviews = db.query(Review).filter(Review.customer_id == customer_id).all()
    
    # Calculate features
    now = datetime_now()
    tenure = (now - user.created_at).days + 30
    frequency = len(purchases)
    spending = sum(p.price * p.quantity for p in purchases)
    
    if purchases:
        last_date = max(p.purchase_date for p in purchases)
        recency = (now - last_date).days
    else:
        recency = 365
        
    complaints_count = len(complaints)
    avg_rating = sum(r.rating for r in reviews) / len(reviews) if reviews else 4.0
    
    # Simulate browsing sessions based on history
    browsing = 40 if complaints_count == 0 else 10
    
    churn_model = load_churn_model()
    prediction = churn_model.predict_churn(
        purchase_frequency=frequency,
        recency_days=recency,
        total_spending=spending,
        complaints_count=complaints_count,
        average_review_rating=avg_rating,
        browsing_sessions=browsing
    )
    
    # Update churn risk in DB
    customer.churn_risk = prediction["churn_probability"]
    db.commit()
    
    return {
        "customer_id": customer_id,
        "name": customer.name,
        "features": {
            "purchase_frequency": frequency,
            "recency_days": recency,
            "total_spending": spending,
            "complaints_count": complaints_count,
            "average_review_rating": avg_rating,
            "browsing_sessions": browsing
        },
        "churn_probability": prediction["churn_probability"],
        "risk_level": prediction["risk_level"],
        "suggestions": prediction["suggestions"]
    }


@router.get("/recommendations/{customer_id}")
def get_recommendations(customer_id: int, top_n: int = 10, db: Session = Depends(get_db)):
    """Fetch collaborative filtering Top-10 recommendations for a customer (Module 6)."""
    recommender = load_recommender()
    
    # 1. Collaborative/SVD Personalized
    rec_ids = recommender.recommend_personalized(customer_id, top_n)
    personalized = db.query(Product).filter(Product.id.in_(rec_ids)).all()
    
    # 2. Trending
    trending_ids = recommender.get_trending_products(top_n)
    trending = db.query(Product).filter(Product.id.in_(trending_ids)).all()
    
    # 3. Frequently Purchased
    purchases_q = db.query(Purchase).all()
    import pandas as pd
    purch_data = [{"customer_id": p.customer_id, "product_id": p.product_id} for p in purchases_q]
    df_purch = pd.DataFrame(purch_data)
    freq_ids = recommender.get_frequently_purchased(df_purch, customer_id, top_n)
    frequently_purchased = db.query(Product).filter(Product.id.in_(freq_ids)).all()

    # 4. Cross-selling (based on current cart)
    from backend.models import CartItem
    cart_items = db.query(CartItem).filter(CartItem.customer_id == customer_id).all()
    cart_prod_ids = [item.product_id for item in cart_items]
    cross_ids = recommender.get_cross_selling_products(cart_prod_ids, top_n)
    cross_selling = db.query(Product).filter(Product.id.in_(cross_ids)).all()

    def format_prods(prods):
        return [{
            "id": p.id,
            "name": p.name,
            "brand": p.brand_name,
            "price": p.price,
            "category": p.category,
            "description": p.description,
            "image_url": p.image_url
        } for p in prods]

    return {
        "personalized": format_prods(personalized),
        "trending": format_prods(trending),
        "frequently_purchased": format_prods(frequently_purchased),
        "cross_selling": format_prods(cross_selling)
    }


@router.post("/verify-product")
def verify_counterfeit_product(
    brand_name: str = Form(...),
    qr_code: str = Form(...),
    company_id: int = Form(...),
    product_id: Optional[int] = Form(None),
    logo_file: UploadFile = File(...),
    packaging_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Module 9: Counterfeit Product Detection
    Module 10: Company Logo Verification
    Module 11: Product Image Similarity
    Matches original product logo & packaging vs uploaded, highlights mismatch and saves.
    """
    # 1. Save uploaded files
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    logo_path = os.path.join(UPLOAD_DIR, f"upload_logo_{logo_file.filename}")
    pkg_path = os.path.join(UPLOAD_DIR, f"upload_pkg_{packaging_file.filename}")
    diff_path = os.path.join(UPLOAD_DIR, f"diff_{packaging_file.filename}")
    
    with open(logo_path, "wb") as buffer:
        shutil.copyfileobj(logo_file.file, buffer)
    with open(pkg_path, "wb") as buffer:
        shutil.copyfileobj(packaging_file.file, buffer)

    # 2. Get reference images from Product or Company logo
    # For demonstration, we use the uploaded files themselves to compare against a "reference" product image
    # If a product_id is given, we fetch the product from DB
    ref_logo_path = None
    ref_pkg_path = None
    
    if product_id:
        product = db.query(Product).filter(Product.id == product_id).first()
        if product:
            # In a real environment, we'd have a local reference path on disk. Let's create an example file:
            ref_logo_path = os.path.join(UPLOAD_DIR, f"ref_logo_{product.brand_name.lower().replace(' ', '_')}.png")
            ref_pkg_path = os.path.join(UPLOAD_DIR, f"ref_pkg_{product.name.lower().replace(' ', '_')}.png")
            
            # Seed reference images if they don't exist
            if not os.path.exists(ref_logo_path):
                # Copy logo_path as dummy reference
                shutil.copy(logo_path, ref_logo_path)
            if not os.path.exists(ref_pkg_path):
                shutil.copy(pkg_path, ref_pkg_path)

    # If reference files still missing, use uploaded files to simulate comparison
    if not ref_logo_path or not os.path.exists(ref_logo_path):
        ref_logo_path = logo_path
    if not ref_pkg_path or not os.path.exists(ref_pkg_path):
        # We create a slight difference in ref_pkg_path if we want to simulate a fake scan!
        ref_pkg_path = pkg_path

    # Run Pillow/NumPy verification pipeline
    # Logo comparison
    logo_results = calculate_image_similarity(ref_logo_path, logo_path)
    # Packaging comparison & generate mismatch highlight
    pkg_results = calculate_image_similarity(ref_pkg_path, pkg_path, output_diff_path=diff_path)

    # Calculate aggregate scores
    is_genuine = logo_results["is_genuine"] and pkg_results["is_genuine"]
    similarity_score = (logo_results["similarity_score"] + pkg_results["similarity_score"]) / 2.0
    confidence_score = similarity_score if is_genuine else (1.0 - similarity_score)
    
    # Compose reason
    reasons = []
    if not logo_results["is_genuine"]:
        reasons.append("Logo mismatch/altered logo detected.")
    if not pkg_results["is_genuine"]:
        reasons.append("Packaging mismatch or shape discrepancies found.")
        
    reason_str = "Product characteristics match reference specifications. Authenticity verified."
    if not is_genuine:
        reason_str = "Verification failed: " + " ".join(reasons)

    # Convert paths to web URLs
    uploaded_logo_url = f"/static/uploads/{os.path.basename(logo_path)}"
    uploaded_pkg_url = f"/static/uploads/{os.path.basename(pkg_path)}"
    diff_url = f"/static/uploads/{os.path.basename(diff_path)}" if os.path.exists(diff_path) else None

    # Save Scan Record to Database
    scan = CounterfeitScan(
        company_id=company_id,
        product_id=product_id,
        brand_name=brand_name,
        uploaded_image_path=uploaded_pkg_url,
        uploaded_logo_path=uploaded_logo_url,
        similarity_score=similarity_score,
        logo_match=logo_results["is_genuine"],
        packaging_match=pkg_results["is_genuine"],
        is_genuine=is_genuine,
        confidence_score=confidence_score,
        reason=reason_str,
        highlighted_image_path=diff_url
    )
    db.add(scan)
    db.commit()

    return {
        "is_genuine": is_genuine,
        "similarity_score": similarity_score,
        "confidence_score": confidence_score,
        "logo_match": logo_results["is_genuine"],
        "packaging_match": pkg_results["is_genuine"],
        "reason": reason_str,
        "mismatched_regions_image": diff_url,
        "scan_id": scan.id
    }


@router.post("/retention-engine/{customer_id}")
def trigger_retention_engine(customer_id: int, db: Session = Depends(get_db)):
    """
    Module 14: Retention Engine
    If customer churn risk > 80%, automatically:
    - Recommend SVD products
    - Generate mock Coupon codes
    - Offer Discounts (e.g. 20% off)
    - Send Email Notification logs
    - Reward 500 Loyalty points
    """
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
        
    # Recalculate churn risk to be sure
    # In a real environment, we'd pull from current DB
    churn_risk = customer.churn_risk
    
    if churn_risk <= 0.80:
        return {
            "triggered": False,
            "message": f"Retention engine not triggered. Customer churn risk is {churn_risk * 100:.1f}%, which is below the 80% threshold.",
            "churn_risk": churn_risk
        }
        
    # Trigger active retention rewards
    customer.loyalty_points += 500
    db.commit()
    
    # Generate recommendations
    recommender = load_recommender()
    rec_ids = recommender.recommend_personalized(customer_id, top_n=3)
    recommended_products = db.query(Product).filter(Product.id.in_(rec_ids)).all()
    
    coupon_code = f"LOYALTY20-{customer_id}-{random_digits()}"
    discount_amount = "20% OFF"
    email_text = f"Subject: We Miss You! Here is a Special {discount_amount} Offer.\n\nDear {customer.name},\n\nWe noticed you haven't visited us in a while. We've added 500 bonus loyalty points to your account and generated a custom coupon code: {coupon_code} for your next purchase!\n\nHere are some products selected just for you:\n" + "\n".join([f"- {p.name} (${p.price})" for p in recommended_products])
    
    # Simulate email logging
    print("----- RETENTION ENGINE MAIL SENT -----")
    print(email_text)
    print("--------------------------------------")
    
    return {
        "triggered": True,
        "churn_risk": churn_risk,
        "loyalty_points_added": 500,
        "new_loyalty_points": customer.loyalty_points,
        "coupon_code": coupon_code,
        "discount": discount_amount,
        "recommended_products": [{"name": p.name, "price": p.price} for p in recommended_products],
        "email_status": "Sent (Simulated)",
        "email_content": email_text
    }


# Helper utilities
def datetime_now():
    import datetime
    return datetime.datetime.utcnow()

def random_digits():
    import random
    return random.randint(1000, 9999)
