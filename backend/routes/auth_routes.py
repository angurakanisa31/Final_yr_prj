from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import uuid

from backend.database import get_db
from backend.models import User, Customer, Company
from backend.auth import get_password_hash, verify_password, create_access_token

router = APIRouter(prefix="/api/auth", tags=["User Authentication"])

class RegisterCustomerSchema(BaseModel):
    email: str
    password: str
    name: str
    phone: Optional[str] = None
    address: Optional[str] = None

class RegisterCompanySchema(BaseModel):
    email: str
    password: str
    name: str
    industry: Optional[str] = None

class LoginSchema(BaseModel):
    email: str
    password: str

class ForgotPasswordSchema(BaseModel):
    email: str



@router.post("/register/customer")
def register_customer(schema: RegisterCustomerSchema, db: Session = Depends(get_db)):
    # Check if email exists
    existing = db.query(User).filter(User.email == schema.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email is already registered")
        
    verification_token = str(uuid.uuid4())
    
    new_user = User(
        email=schema.email,
        hashed_password=get_password_hash(schema.password),
        role="customer",
        is_verified=True,  # Auto-verify for ease of local demo
        verification_token=verification_token
    )
    db.add(new_user)
    db.flush()  # Retrieve User ID
    
    customer = Customer(
        id=new_user.id,
        name=schema.name,
        phone=schema.phone,
        address=schema.address,
        loyalty_points=100,  # 100 welcome loyalty points
        spending_score=50.0,
        churn_risk=0.0
    )
    db.add(customer)
    db.commit()
    
    return {"message": "Customer registered successfully. Welcome points credited!", "is_verified": True}


@router.post("/register/company")
def register_company(schema: RegisterCompanySchema, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == schema.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email is already registered")
        
    verification_token = str(uuid.uuid4())
    
    new_user = User(
        email=schema.email,
        hashed_password=get_password_hash(schema.password),
        role="company",
        is_verified=True,
        verification_token=verification_token
    )
    db.add(new_user)
    db.flush()
    
    company = Company(
        id=new_user.id,
        name=schema.name,
        industry=schema.industry,
        verification_status="Approved"
    )
    db.add(company)
    db.commit()
    
    return {"message": "Company registered successfully.", "is_verified": True}


@router.post("/login")
def login(schema: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == schema.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
        
    if not verify_password(schema.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
        
    # Generate Token
    access_token = create_access_token(data={"sub": user.id, "role": user.role})
    
    # Get Profile Name
    name = "User"
    company_id = None
    customer_id = None
    if user.role == "customer":
        cust = db.query(Customer).filter(Customer.id == user.id).first()
        if cust:
            name = cust.name
            customer_id = cust.id
    elif user.role == "company":
        comp = db.query(Company).filter(Company.id == user.id).first()
        if comp:
            name = comp.name
            company_id = comp.id
            
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "role": user.role,
            "name": name,
            "company_id": company_id,
            "customer_id": customer_id
        }
    }


@router.post("/forgot-password")
def forgot_password(schema: ForgotPasswordSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == schema.email).first()
    if not user:
        # Prevent user enumeration, return generic success
        return {"message": "If the account exists, a password reset link has been sent."}
        
    reset_token = str(uuid.uuid4())
    user.reset_token = reset_token
    db.commit()
    
    print(f"----- PASSWORD RESET EMAIL -----")
    print(f"Reset Link: http://localhost:5173/reset-password?token={reset_token}")
    print("---------------------------------")
    
    return {"message": "If the account exists, a password reset link has been sent."}


@router.get("/verify-email")
def verify_email(token: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.verification_token == token).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid verification token")
    user.is_verified = True
    user.verification_token = None
    db.commit()
    return {"message": "Email verified successfully"}
