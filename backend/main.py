from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import os

from backend.database import engine, Base, get_db
from backend.models import Product
from backend.routes import auth_routes, customer, ml_endpoints, analytics, reports, chatbot

# Initialize FastAPI App
app = FastAPI(
    title="AI-Powered Integrated E-commerce & CRM Framework",
    description="Backend services for customer analytics, churn prediction, product recommendation, and counterfeit detection",
    version="1.0.0"
)

# CORS Configuration for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure database tables exist
Base.metadata.create_all(bind=engine)

# Create static directories for media uploads
os.makedirs("backend/static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# Include Modular Routers
app.include_router(auth_routes.router)
app.include_router(customer.router)
app.include_router(ml_endpoints.router)
app.include_router(analytics.router)
app.include_router(reports.router)
app.include_router(chatbot.router)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "project": "AI-Powered Integrated E-commerce & CRM Framework",
        "documentation": "/docs"
    }


@app.get("/api/products")
def list_products(category: str = Query(None), q: str = Query(None), db: Session = Depends(get_db)):
    """Fetch product catalog with optional category or keyword query filters."""
    query = db.query(Product)
    if category:
        query = query.filter(Product.category == category)
    if q:
        query = query.filter((Product.name.like(f"%{q}%")) | (Product.brand_name.like(f"%{q}%")))
        
    products = query.all()
    return [{
        "id": p.id,
        "name": p.name,
        "brand": p.brand_name,
        "price": p.price,
        "category": p.category,
        "description": p.description,
        "image_url": p.image_url,
        "stock": p.stock
    } for p in products]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
