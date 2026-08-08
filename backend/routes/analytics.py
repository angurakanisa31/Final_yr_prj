from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
import datetime
from backend.database import get_db
from backend.models import User, Customer, Product, Purchase, Review, Complaint, CounterfeitScan, Company

router = APIRouter(prefix="/api/analytics", tags=["Dashboards & Analytics"])

@router.get("/customer/{customer_id}")
def get_customer_dashboard_data(customer_id: int, db: Session = Depends(get_db)):
    """Module 16: Customer Dashboard metrics."""
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
        
    purchases = db.query(Purchase).filter(Purchase.customer_id == customer_id).all()
    reviews = db.query(Review).filter(Review.customer_id == customer_id).all()
    complaints = db.query(Complaint).filter(Complaint.complaint_id == customer_id if hasattr(Complaint, 'complaint_id') else Complaint.customer_id == customer_id).all()

    total_spent = sum(p.price * p.quantity for p in purchases)
    purchase_count = len(purchases)
    
    # Calculate favorite category
    category_counts = {}
    for p in purchases:
        product = db.query(Product).filter(Product.id == p.product_id).first()
        if product:
            category_counts[product.category] = category_counts.get(product.category, 0) + 1
            
    fav_cat = max(category_counts, key=category_counts.get) if category_counts else "None"

    return {
        "customer_id": customer_id,
        "name": customer.name,
        "loyalty_points": customer.loyalty_points,
        "segment": customer.segment,
        "churn_risk": customer.churn_risk,
        "clv_value": customer.clv_value,
        "total_spent": total_spent,
        "purchase_count": purchase_count,
        "favorite_category": fav_cat,
        "reviews_submitted": len(reviews),
        "complaints_registered": len(complaints)
    }

@router.get("/company")
def get_company_dashboard_data(db: Session = Depends(get_db)):
    """Module 16: Company Dashboard analytics."""
    customers = db.query(Customer).all()
    products = db.query(Product).all()
    purchases = db.query(Purchase).all()
    scans = db.query(CounterfeitScan).all()

    total_customers = len(customers)
    total_products = len(products)
    total_revenue = sum(p.price * p.quantity for p in purchases)
    
    # Customer segments counts
    segments = {"Premium": 0, "Regular Customer": 0, "New Customer": 0, "High-Risk": 0}
    for c in customers:
        seg = c.segment
        if seg in segments:
            segments[seg] += 1
        elif "Regular" in seg:
            segments["Regular Customer"] += 1
        elif "New" in seg:
            segments["New Customer"] += 1
        else:
            segments["High-Risk"] += 1

    # Churn distribution
    churn_dist = {"Low Risk (<30%)": 0, "Medium Risk (30-80%)": 0, "High Risk (>80%)": 0}
    for c in customers:
        risk = c.churn_risk
        if risk < 0.3:
            churn_dist["Low Risk (<30%)"] += 1
        elif risk < 0.8:
            churn_dist["Medium Risk (30-80%)"] += 1
        else:
            churn_dist["High Risk (>80%)"] += 1

    # Monthly Sales Trend
    # Group purchases by month
    monthly_sales = {}
    for p in purchases:
        month = p.purchase_date.strftime("%b %Y")  # e.g., "Aug 2026"
        monthly_sales[month] = monthly_sales.get(month, 0.0) + (p.price * p.quantity)
        
    # Sort monthly sales chronologically
    sorted_months = sorted(monthly_sales.keys(), key=lambda x: datetime.datetime.strptime(x, "%b %Y"))
    sales_trend = [{"month": m, "sales": monthly_sales[m]} for m in sorted_months]

    # Category Revenue Distribution
    category_revenue = {}
    for p in purchases:
        prod = db.query(Product).filter(Product.id == p.product_id).first()
        if prod:
            category_revenue[prod.category] = category_revenue.get(prod.category, 0.0) + (p.price * p.quantity)
    cat_distribution = [{"category": k, "revenue": v} for k, v in category_revenue.items()]

    # Best-selling products (frequency)
    product_frequency = {}
    for p in purchases:
        product_frequency[p.product_id] = product_frequency.get(p.product_id, 0) + p.quantity
        
    sorted_prods = sorted(product_frequency.items(), key=lambda x: x[1], reverse=True)
    best_selling = []
    slow_moving = []
    
    # Top 5 best selling
    for pid, freq in sorted_prods[:5]:
        prod = db.query(Product).filter(Product.id == pid).first()
        if prod:
            best_selling.append({"name": prod.name, "brand": prod.brand_name, "quantity": freq, "revenue": prod.price * freq})
            
    # Bottom 5 slow moving (excluding best sellers, or simply least sold)
    for pid, freq in sorted_prods[-5:]:
        prod = db.query(Product).filter(Product.id == pid).first()
        if prod:
            slow_moving.append({"name": prod.name, "brand": prod.brand_name, "quantity": freq, "stock": prod.stock})

    # Fake product alerts summary
    fake_scans_count = sum(1 for s in scans if not s.is_genuine)
    total_scans_count = len(scans)
    recent_fake_alerts = []
    for s in scans:
        if not s.is_genuine:
            recent_fake_alerts.append({
                "id": s.id,
                "brand_name": s.brand_name,
                "similarity_score": s.similarity_score,
                "reason": s.reason,
                "date": s.scan_date.strftime("%Y-%m-%d %H:%M:%S")
            })

    # High-level metrics
    # Churn Rate = (High Risk count / Total Customers) * 100
    high_risk_count = churn_dist["High Risk (>80%)"]
    retention_rate = ((total_customers - high_risk_count) / total_customers * 100) if total_customers > 0 else 100.0
    
    # Recommendation Accuracy (simulated based on rating scores: % of positive reviews)
    total_reviews = db.query(Review).count()
    positive_reviews = db.query(Review).filter(Review.rating >= 4.0).count()
    rec_accuracy = (positive_reviews / total_reviews * 100) if total_reviews > 0 else 85.0

    return {
        "metrics": {
            "total_revenue": total_revenue,
            "total_customers": total_customers,
            "total_products": total_products,
            "retention_rate": retention_rate,
            "recommendation_accuracy": rec_accuracy,
            "fake_scans_count": fake_scans_count,
            "total_scans_count": total_scans_count
        },
        "segments": segments,
        "churn_risk_distribution": churn_dist,
        "sales_trend": sales_trend,
        "category_distribution": cat_distribution,
        "best_selling_products": best_selling,
        "slow_moving_products": slow_moving,
        "recent_fake_alerts": recent_fake_alerts[:5]
    }
