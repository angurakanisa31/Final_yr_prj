import os
import random
import datetime
import pickle
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from backend.database import SessionLocal, engine, Base
from backend.models import User, Customer, Company, Product, Purchase, Review, Complaint, CartItem, WishlistItem
from backend.auth import get_password_hash
from backend.ml.segmentation import CustomerSegmenter
from backend.ml.churn_prediction import ChurnPredictor
from backend.ml.recommendation import SVDRecommender

# Create models directory if not exists
os.makedirs("backend/ml/models", exist_ok=True)

def generate_mock_data():
    print("Initializing Database...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    print("Generating Users, Customers and Companies...")
    # Admin User
    admin = User(
        email="admin@ecommerce.com",
        hashed_password=get_password_hash("admin123"),
        role="admin",
        is_verified=True
    )
    db.add(admin)

    # Companies
    companies_data = [
        {"name": "Apple Inc.", "email": "apple@company.com", "industry": "Consumer Electronics", "logo": "apple_logo.png"},
        {"name": "Nike", "email": "nike@company.com", "industry": "Sportswear", "logo": "nike_logo.png"},
        {"name": "Sony", "email": "sony@company.com", "industry": "Entertainment & Tech", "logo": "sony_logo.png"},
        {"name": "Adidas", "email": "adidas@company.com", "industry": "Athletic Wear", "logo": "adidas_logo.png"}
    ]
    
    company_instances = []
    for idx, c in enumerate(companies_data):
        user_c = User(
            email=c["email"],
            hashed_password=get_password_hash("company123"),
            role="company",
            is_verified=True
        )
        db.add(user_c)
        db.flush()  # Get ID
        
        comp = Company(
            id=user_c.id,
            name=c["name"],
            logo_url=c["logo"],
            industry=c["industry"],
            verification_status="Approved"
        )
        db.add(comp)
        company_instances.append(comp)

    # Customers
    names = [
        "James Smith", "Michael Smith", "Robert Smith", "Maria Garcia", "David Smith", 
        "Maria Rodriguez", "Mary Smith", "Maria Hernandez", "Maria Martinez", "James Johnson",
        "John Smith", "William Smith", "Patricia Smith", "Robert Johnson", "Linda Smith",
        "John Johnson", "David Johnson", "Elizabeth Smith", "Barbara Smith", "Richard Smith",
        "Thomas Smith", "Joseph Smith", "Jennifer Smith", "Charles Smith", "Christopher Smith",
        "Daniel Smith", "Matthew Smith", "Sarah Smith", "Jessica Smith", "Mark Smith"
    ]
    
    customer_instances = []
    for idx, name in enumerate(names):
        email = f"{name.lower().replace(' ', '.')}@gmail.com"
        user_cust = User(
            email=email,
            hashed_password=get_password_hash("customer123"),
            role="customer",
            is_verified=True
        )
        db.add(user_cust)
        db.flush()
        
        cust = Customer(
            id=user_cust.id,
            name=name,
            phone=f"+1 555-01{idx:02d}",
            address=f"{100 + idx} Main Street, Tech City, USA",
            loyalty_points=random.randint(50, 2500),
            spending_score=random.uniform(10, 100)
        )
        db.add(cust)
        customer_instances.append(cust)

    print("Generating Products...")
    categories = ["Electronics", "Apparel", "Footwear", "Accessories"]
    products_pool = [
        # Apple Electronics
        {"name": "iPhone 15 Pro", "brand": "Apple Inc.", "price": 999.0, "category": "Electronics", "desc": "Titanium design, A17 Pro chip, Action button."},
        {"name": "MacBook Air M3", "brand": "Apple Inc.", "price": 1099.0, "category": "Electronics", "desc": "Thin and light laptop with M3 chip and liquid retina display."},
        {"name": "Apple Watch Series 9", "brand": "Apple Inc.", "price": 399.0, "category": "Electronics", "desc": "Advanced health sensors, bright display, double tap gesture."},
        # Nike Apparel/Footwear
        {"name": "Air Max 270", "brand": "Nike", "price": 160.0, "category": "Footwear", "desc": "Nike's first lifestyle Air Max brings you style and comfort."},
        {"name": "Dri-FIT Training Shirt", "brand": "Nike", "price": 35.0, "category": "Apparel", "desc": "Sweat-wicking fabric helps keep you dry and comfortable."},
        {"name": "Nike Tech Fleece Hoodie", "brand": "Nike", "price": 130.0, "category": "Apparel", "desc": "Premium lightweight fleece that is smooth both inside and out."},
        # Sony Accessories/Electronics
        {"name": "WH-1000XM5 Headphones", "brand": "Sony", "price": 398.0, "category": "Electronics", "desc": "Industry leading noise canceling wireless headphones."},
        {"name": "PlayStation 5 Console", "brand": "Sony", "price": 499.0, "category": "Electronics", "desc": "Experience lightning-fast loading with an ultra-high speed SSD."},
        # Adidas Apparel/Footwear
        {"name": "Ultraboost Light", "brand": "Adidas", "price": 190.0, "category": "Footwear", "desc": "Epic energy in the lightest Ultraboost ever."},
        {"name": "Adidas Originals Track Jacket", "brand": "Adidas", "price": 85.0, "category": "Apparel", "desc": "Classic sporty track jacket with iconic 3-stripes details."}
    ]

    product_instances = []
    for p in products_pool:
        # Find matching company
        company = [c for c in company_instances if c.name == p["brand"]][0]
        prod = Product(
            company_id=company.id,
            name=p["name"],
            brand_name=p["brand"],
            price=p["price"],
            category=p["category"],
            description=p["desc"],
            stock=random.randint(20, 150),
            logo_url=f"{p['brand'].lower().replace(' ', '_')}_logo.png",
            image_url=f"{p['name'].lower().replace(' ', '_')}.png"
        )
        db.add(prod)
        db.flush()
        product_instances.append(prod)

    print("Generating Purchases, Reviews, Cart and Wishlist items...")
    # Add purchases and reviews for SVD recommender and churn features
    purchases_list = []
    reviews_list = []
    
    # Pre-populate dates within past year
    now = datetime.datetime.utcnow()
    
    # We want to create some churned vs active customer profiles
    # Churned: last purchase long ago, high complaints, low ratings
    # Active: recent purchases, frequent visits, high ratings
    
    for idx, cust in enumerate(customer_instances):
        # Determine behavior profile
        is_churn_profile = (idx % 4 == 0)  # ~25% churn profiles
        
        num_purchases = random.randint(1, 3) if is_churn_profile else random.randint(5, 15)
        last_purchase_days_ago = random.randint(100, 360) if is_churn_profile else random.randint(2, 45)
        
        # Draw products
        bought_prods = random.sample(product_instances, k=min(num_purchases, len(product_instances)))
        
        for p_idx, prod in enumerate(bought_prods):
            # Calculate purchase date
            days_ago = last_purchase_days_ago + p_idx * random.randint(5, 30) if is_churn_profile else random.randint(2, last_purchase_days_ago + 1)
            purchase_date = now - datetime.timedelta(days=days_ago)
            
            purchase = Purchase(
                customer_id=cust.id,
                product_id=prod.id,
                quantity=random.randint(1, 2),
                price=prod.price,
                purchase_date=purchase_date,
                seasonal_trend=random.choice(["Winter", "Summer", "Spring", "Autumn", "General"])
            )
            db.add(purchase)
            purchases_list.append(purchase)
            
            # Review
            if random.random() > 0.3:
                rating = random.choice([1.0, 2.0, 3.0]) if is_churn_profile else random.choice([4.0, 5.0])
                sentiment_map = {1.0: ("Negative", 0.1), 2.0: ("Negative", 0.3), 3.0: ("Neutral", 0.5), 4.0: ("Positive", 0.85), 5.0: ("Positive", 0.98)}
                sentiment, score = sentiment_map[rating]
                
                review = Review(
                    customer_id=cust.id,
                    product_id=prod.id,
                    rating=rating,
                    review_text=f"Sample review text for product {prod.name}. The rating represents my opinion.",
                    sentiment=sentiment,
                    sentiment_score=score,
                    review_date=purchase_date + datetime.timedelta(days=random.randint(1, 5))
                )
                db.add(review)
                reviews_list.append(review)
                
        # Complaints
        if is_churn_profile and random.random() > 0.4:
            complaint = Complaint(
                customer_id=cust.id,
                complaint_text="My order was delayed, and the customer support is slow to respond. Dissatisfied.",
                status="Open",
                complaint_date=now - datetime.timedelta(days=last_purchase_days_ago - 5)
            )
            db.add(complaint)

        # Cart / Wishlist
        if not is_churn_profile:
            # Active users have items in cart
            cart_prod = random.choice(product_instances)
            cart = CartItem(
                customer_id=cust.id,
                product_id=cart_prod.id,
                quantity=random.randint(1, 3)
            )
            db.add(cart)
            
            wish_prod = random.choice(product_instances)
            wish = WishlistItem(
                customer_id=cust.id,
                product_id=wish_prod.id
            )
            db.add(wish)

    db.commit()
    print("Database seeding completed.")
    return db

def train_and_save_models(db: Session):
    print("Training ML Models...")
    
    # 1. Gather Customer Data for K-Means and XGBoost
    customers = db.query(Customer).all()
    now = datetime.datetime.utcnow()
    
    customer_features = []
    
    for c in customers:
        # Calculate features:
        # Recency: days since last purchase
        # Frequency: total purchases count
        # Monetary: total spending
        # Tenure: days since user registration
        
        purchases = db.query(Purchase).filter(Purchase.customer_id == c.id).all()
        user = db.query(User).filter(User.id == c.id).first()
        complaints = db.query(Complaint).filter(Complaint.customer_id == c.id).all()
        reviews = db.query(Review).filter(Review.customer_id == c.id).all()
        
        tenure = (now - user.created_at).days + 30  # add buffer so not 0
        frequency = len(purchases)
        monetary = sum(p.price * p.quantity for p in purchases)
        
        if purchases:
            last_date = max(p.purchase_date for p in purchases)
            recency = (now - last_date).days
        else:
            recency = 365  # high recency if no purchase
            
        complaints_count = len(complaints)
        avg_rating = np.mean([r.rating for r in reviews]) if reviews else 4.0
        
        # Simulate browsing sessions based on churn behavior
        # Churn profiles have lower browsing or high bouncing sessions
        is_churn = 1 if (recency > 90 or complaints_count > 0) else 0
        browsing = random.randint(2, 8) if is_churn else random.randint(15, 60)
        
        customer_features.append({
            "customer_id": c.id,
            "recency": recency,
            "recency_days": recency,
            "frequency": frequency,
            "purchase_frequency": frequency,
            "monetary_value": monetary,
            "total_spending": monetary,
            "tenure": tenure,
            "complaints_count": complaints_count,
            "average_review_rating": avg_rating,
            "browsing_sessions": browsing,
            "churned": is_churn
        })
        
    df_features = pd.DataFrame(customer_features)
    
    # ---- 1. Customer Segmentation (K-Means) ----
    print("Fitting K-Means Segmentation...")
    segmenter = CustomerSegmenter(n_clusters=4)
    segmenter.fit(df_features)
    
    # Save Segmenter
    with open("backend/ml/models/segmenter.pkl", "wb") as f:
        pickle.dump(segmenter, f)
        
    # Update customer table with predicted segments
    predicted_segments = segmenter.predict(df_features)
    for idx, c_feat in enumerate(customer_features):
        c_id = c_feat["customer_id"]
        cust = db.query(Customer).filter(Customer.id == c_id).first()
        cust.segment = predicted_segments[idx]
        
        # Simple rule-based CLV prediction for database
        spend = c_feat["monetary_value"]
        if spend > 1500:
            cust.clv_value = "High Value"
        elif spend > 400:
            cust.clv_value = "Medium Value"
        else:
            cust.clv_value = "Low Value"
    
    # ---- 2. Churn Prediction (XGBoost) ----
    print("Fitting XGBoost Churn Predictor...")
    # Duplicate features to have a larger set for training stability
    large_features = pd.concat([df_features] * 20, ignore_index=True)
    # Add slight random noise to numeric columns to simulate variance
    for col in ["recency_days", "purchase_frequency", "total_spending", "average_review_rating", "browsing_sessions"]:
        large_features[col] = large_features[col] + np.random.normal(0, large_features[col].std() * 0.05, len(large_features))
        large_features[col] = large_features[col].clip(lower=0)
    # Average rating clips between 1 and 5
    large_features["average_review_rating"] = large_features["average_review_rating"].clip(1, 5)

    predictor = ChurnPredictor()
    metrics = predictor.fit_and_evaluate(large_features)
    print(f"XGBoost Churn Predictor Metrics: {metrics}")
    
    # Save Churn Model
    predictor.save("backend/ml/models/churn_model.pkl")

    # Update customer table with predicted churn risks
    for c_feat in customer_features:
        c_id = c_feat["customer_id"]
        pred_res = predictor.predict_churn(
            purchase_frequency=c_feat["purchase_frequency"],
            recency_days=c_feat["recency_days"],
            total_spending=c_feat["total_spending"],
            complaints_count=c_feat["complaints_count"],
            average_review_rating=c_feat["average_review_rating"],
            browsing_sessions=c_feat["browsing_sessions"]
        )
        cust = db.query(Customer).filter(Customer.id == c_id).first()
        cust.churn_risk = pred_res["churn_probability"]

    # ---- 3. Recommendation Engine (SVD) ----
    print("Fitting SVD Collaborative Filter...")
    # Gather reviews/ratings
    reviews_db = db.query(Review).all()
    ratings_data = [{"customer_id": r.customer_id, "product_id": r.product_id, "rating": r.rating} for r in reviews_db]
    df_ratings = pd.DataFrame(ratings_data)
    
    products_db = db.query(Product).all()
    products_data = [{"id": p.id, "name": p.name, "category": p.category, "brand_name": p.brand_name} for p in products_db]
    df_products = pd.DataFrame(products_data)
    
    recommender = SVDRecommender(n_factors=5)
    recommender.fit(df_ratings, df_products)
    
    # Save Recommender
    with open("backend/ml/models/recommender.pkl", "wb") as f:
        pickle.dump(recommender, f)
        
    db.commit()
    db.close()
    print("All ML models trained and saved to backend/ml/models/")

if __name__ == "__main__":
    db_session = generate_mock_data()
    train_and_save_models(db_session)
