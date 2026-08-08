from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
from backend.database import get_db
from backend.models import User, Customer, Product, Purchase, CartItem, WishlistItem, Review
from backend.auth import get_current_user
from backend.ml.sentiment import analyze_sentiment

router = APIRouter(prefix="/api/customer", tags=["Customer Data Management"])

@router.get("/profile")
def get_customer_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "customer":
        raise HTTPException(status_code=400, detail="User is not a customer")
    customer = db.query(Customer).filter(Customer.id == current_user.id).first()
    if not customer:
        raise HTTPException(status_code=44, detail="Customer profile not found")
    return {
        "email": current_user.email,
        "name": customer.name,
        "phone": customer.phone,
        "address": customer.address,
        "loyalty_points": customer.loyalty_points,
        "segment": customer.segment,
        "clv_value": customer.clv_value,
        "churn_risk": customer.churn_risk
    }

@router.put("/profile")
def update_customer_profile(name: str, phone: Optional[str] = None, address: Optional[str] = None, 
                            current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "customer":
        raise HTTPException(status_code=400, detail="User is not a customer")
    customer = db.query(Customer).filter(Customer.id == current_user.id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer profile not found")
    customer.name = name
    if phone is not None:
        customer.phone = phone
    if address is not None:
        customer.address = address
    db.commit()
    return {"message": "Profile updated successfully"}

@router.get("/purchases")
def get_purchase_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "customer":
        raise HTTPException(status_code=400, detail="User is not a customer")
    purchases = db.query(Purchase).filter(Purchase.customer_id == current_user.id).all()
    
    result = []
    for p in purchases:
        product = db.query(Product).filter(Product.id == p.product_id).first()
        result.append({
            "purchase_id": p.id,
            "product_id": p.product_id,
            "product_name": product.name if product else "Unknown Product",
            "brand": product.brand_name if product else "Unknown",
            "quantity": p.quantity,
            "price": p.price,
            "total": p.price * p.quantity,
            "date": p.purchase_date.strftime("%Y-%m-%d %H:%M:%S")
        })
    return result

@router.get("/cart")
def get_cart_items(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "customer":
        raise HTTPException(status_code=400, detail="User is not a customer")
    items = db.query(CartItem).filter(CartItem.customer_id == current_user.id).all()
    result = []
    for item in items:
        product = db.query(Product).filter(Product.id == item.product_id).first()
        if product:
            result.append({
                "cart_item_id": item.id,
                "product_id": product.id,
                "name": product.name,
                "price": product.price,
                "quantity": item.quantity,
                "total": product.price * item.quantity,
                "image_url": product.image_url
            })
    return result

@router.post("/cart/add")
def add_to_cart(product_id: int, quantity: int = 1, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "customer":
        raise HTTPException(status_code=400, detail="User is not a customer")
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
        
    # Check if already in cart
    existing = db.query(CartItem).filter(CartItem.customer_id == current_user.id, CartItem.product_id == product_id).first()
    if existing:
        existing.quantity += quantity
    else:
        new_item = CartItem(customer_id=current_user.id, product_id=product_id, quantity=quantity)
        db.add(new_item)
    db.commit()
    return {"message": "Added to cart successfully"}

@router.put("/cart/update")
def update_cart_quantity(cart_item_id: int, quantity: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "customer":
        raise HTTPException(status_code=400, detail="User is not a customer")
    item = db.query(CartItem).filter(CartItem.id == cart_item_id, CartItem.customer_id == current_user.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Cart item not found")
    if quantity <= 0:
        db.delete(item)
    else:
        item.quantity = quantity
    db.commit()
    return {"message": "Cart updated"}

@router.delete("/cart/remove/{cart_item_id}")
def remove_from_cart(cart_item_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "customer":
        raise HTTPException(status_code=400, detail="User is not a customer")
    item = db.query(CartItem).filter(CartItem.id == cart_item_id, CartItem.customer_id == current_user.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Cart item not found")
    db.delete(item)
    db.commit()
    return {"message": "Item removed from cart"}

@router.get("/wishlist")
def get_wishlist(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "customer":
        raise HTTPException(status_code=400, detail="User is not a customer")
    items = db.query(WishlistItem).filter(WishlistItem.customer_id == current_user.id).all()
    result = []
    for item in items:
        product = db.query(Product).filter(Product.id == item.product_id).first()
        if product:
            result.append({
                "wishlist_id": item.id,
                "product_id": product.id,
                "name": product.name,
                "price": product.price,
                "brand": product.brand_name,
                "image_url": product.image_url
            })
    return result

@router.post("/wishlist/add")
def add_to_wishlist(product_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "customer":
        raise HTTPException(status_code=400, detail="User is not a customer")
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
        
    existing = db.query(WishlistItem).filter(WishlistItem.customer_id == current_user.id, WishlistItem.product_id == product_id).first()
    if not existing:
        new_item = WishlistItem(customer_id=current_user.id, product_id=product_id)
        db.add(new_item)
        db.commit()
    return {"message": "Added to wishlist"}

@router.delete("/wishlist/remove/{product_id}")
def remove_from_wishlist(product_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "customer":
        raise HTTPException(status_code=400, detail="User is not a customer")
    item = db.query(WishlistItem).filter(WishlistItem.product_id == product_id, WishlistItem.customer_id == current_user.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Wishlist item not found")
    db.delete(item)
    db.commit()
    return {"message": "Removed from wishlist"}

@router.post("/reviews/submit")
def submit_product_review(product_id: int, rating: float, review_text: str, 
                          current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "customer":
        raise HTTPException(status_code=400, detail="User is not a customer")
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
        
    # Analyze Sentiment using BERT (or TextBlob fallback)
    sentiment, sentiment_score = analyze_sentiment(review_text)
    
    review = Review(
        customer_id=current_user.id,
        product_id=product_id,
        rating=rating,
        review_text=review_text,
        sentiment=sentiment,
        sentiment_score=sentiment_score
    )
    db.add(review)
    db.commit()
    
    return {
        "message": "Review submitted successfully",
        "predicted_sentiment": sentiment,
        "sentiment_score": sentiment_score
    }
