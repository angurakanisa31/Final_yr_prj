import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.main import app
from backend.database import Base, get_db
from backend.ml.sentiment import analyze_sentiment
from backend.ml.image_similarity import calculate_image_similarity

# Use a separate test database file
TEST_DATABASE_URL = "sqlite:///c:/Users/angur/OneDrive/Documents/Final Yr Project/backend/test_ecommerce.db"

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override get_db dependency
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    # Remove test DB file
    if os.path.exists("backend/test_ecommerce.db"):
        try:
            os.remove("backend/test_ecommerce.db")
        except Exception:
            pass

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_auth_register_customer():
    response = client.post(
        "/api/auth/register/customer",
        json={
            "email": "test_customer@gmail.com",
            "password": "testpassword123",
            "name": "Test Customer",
            "phone": "+1 555-0100",
            "address": "123 Test Lane"
        }
    )
    assert response.status_code == 200
    assert response.json()["is_verified"] is True

def test_auth_register_company():
    response = client.post(
        "/api/auth/register/company",
        json={
            "email": "test_company@company.com",
            "password": "testpassword123",
            "name": "Test Company Corp",
            "industry": "Software Services"
        }
    )
    assert response.status_code == 200

def test_auth_login():
    response = client.post(
        "/api/auth/login",
        json={
            "email": "test_customer@gmail.com",
            "password": "testpassword123"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["role"] == "customer"
    assert data["user"]["name"] == "Test Customer"

def test_sentiment_analyzer():
    # Test positive text
    sentiment, score = analyze_sentiment("This product is absolutely amazing, I love it!")
    assert sentiment == "Positive"
    assert score >= 0.5

    # Test negative text
    sentiment, score = analyze_sentiment("Worst service ever, broken packaging and slow delivery.")
    assert sentiment == "Negative"
    assert score <= 0.5

def test_image_similarity_engine():
    # Create mock images for testing
    from PIL import Image
    os.makedirs("backend/static/uploads", exist_ok=True)
    
    img1_path = "backend/static/uploads/test_img1.png"
    img2_path = "backend/static/uploads/test_img2.png"
    diff_path = "backend/static/uploads/test_diff.png"

    # Create identical white images
    img1 = Image.new("RGB", (300, 300), color="white")
    img1.save(img1_path)
    img2 = Image.new("RGB", (300, 300), color="white")
    img2.save(img2_path)

    # Compare identical
    results = calculate_image_similarity(img1_path, img2_path, diff_path)
    assert results["is_genuine"] is True
    assert results["similarity_score"] >= 0.95
    assert results["logo_match"] is True

    # Alter img2 with a black block to mock counterfeit
    pixels = img2.load()
    for x in range(100, 200):
        for y in range(100, 200):
            pixels[x, y] = (0, 0, 0)
    img2.save(img2_path)

    # Compare different
    results_fake = calculate_image_similarity(img1_path, img2_path, diff_path)
    assert results_fake["similarity_score"] < results["similarity_score"]
    assert os.path.exists(diff_path)

    # Clean up test files
    for path in [img1_path, img2_path, diff_path]:
        if os.path.exists(path):
            os.remove(path)
