# AI-Powered Integrated E-Commerce & CRM Framework

An intelligent framework for Personalized Product Recommendations (SVD), Customer Churn Prediction (XGBoost), Customer Segmentation (K-Means), Sentiment Analysis (BERT), and Counterfeit Product Detection (Pillow & NumPy Mismatch Engine).

Developed with a **FastAPI backend** (Python 3.12) and a **React.js frontend** (Vite, vanilla CSS glassmorphism styling).

---

## Folder Structure
```
Final Yr Project/
├── backend/
│   ├── main.py                     # Entry point of the FastAPI application
│   ├── database.py                 # SQLite database engine configuration
│   ├── models.py                   # SQLAlchemy database schemas mapping
│   ├── auth.py                     # JWT token handling and bcrypt utilities
│   ├── routes/
│   │   ├── auth_routes.py          # User authentication endpoints (register, login)
│   │   ├── customer.py             # Profile, cart, wishlist, and reviews APIs
│   │   ├── ml_endpoints.py         # Recs, churn predictions, scans, and retention engine
│   │   ├── analytics.py            # Dashboard metrics compiler (sales trend, segments)
│   │   ├── reports.py              # Excel, PDF, and CSV reports generation
│   │   └── chatbot.py              # Semantic AI Assistant dialog parser
│   ├── ml/
│   │   ├── models/                 # Folder storing saved .pkl binaries
│   │   ├── preprocessing.py        # Scaler and label encoder helpers
│   │   ├── segmentation.py         # K-Means customer tier partitioner
│   │   ├── churn_prediction.py     # XGBoost churn score and retention suggestion mapper
│   │   ├── recommendation.py       # Collaborative filtering SVD solver
│   │   ├── sentiment.py            # BERT pipeline review scorer
│   │   ├── image_similarity.py     # Image difference pixel mismatch highlights
│   │   └── train_models.py         # Database seeder and training wrapper
│   └── tests/
│       └── test_api.py             # Pytest automated API testing script
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Single-page interface dashboard app
│   │   ├── index.css               # Design system glassmorphic CSS rules
│   │   └── main.jsx                # React mount entrypoint
│   └── package.json                # React configurations and dependencies
├── docs/
│   ├── ieee_paper.md               # Complete academic IEEE paper
│   ├── project_report.md           # Formal project report manual
│   └── slides.md                   # Presentation slide deck outline
└── README.md                       # Master Readme, Deployment Guide & User Manual
```

---

## Deployment Guide

### Prerequisites
1. **Python 3.12**: Confirm it is available in your terminal.
2. **Node.js (v18+) & NPM**: Required to resolve React modules.

### Step 1: Install Python Dependencies & Seeding
From the root project folder:
```bash
# Verify packages list in Python 3.12 and initialize models
C:\Users\angur\AppData\Local\Programs\Python\Python312\python.exe -m backend.ml.train_models
```
*This command creates the `backend/ecommerce.db` database, seeds it with mock products/purchases/customers, trains the K-Means, XGBoost, and SVD engines, and dumps them to `backend/ml/models/`.*

### Step 2: Start the FastAPI Server
```bash
C:\Users\angur\AppData\Local\Programs\Python\Python312\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
*The API gateway starts running at `http://127.0.0.1:8000`. You can inspect the Swagger interface docs at `http://127.0.0.1:8000/docs`.*

### Step 3: Run the React Frontend
Navigate to the `frontend/` folder:
```bash
cd frontend
npm run dev
```
*The dev client starts running at `http://localhost:5173`. Open this URL in your web browser.*

---

## User Manual

### A. Testing Customer Portal
1. **Log In**: Start the application, choose "Log In", and enter one of the seeded customer accounts:
   * **Email**: `james.smith@gmail.com`
   * **Password**: `customer123`
2. **Catalog Shopping**: Browse catalog products. Use category pill buttons or search terms to narrow items. Add items to your **Cart** or **Wishlist**.
3. **SVD Recommendations**: Scroll to the recommendation panel to inspect SVD customized lists: "Personalized for You" (Collaborative Filter), "Trending Overall" (Cold-start values), and "Frequently Purchased Together" (Cross-selling).
4. **Ratings & BERT Review Sentiment**:
   * Navigate to **Purchase History**.
   * Click **Rate & Review** for a product.
   * Write feedback (e.g. *"Great product, outstanding build and quality!"*).
   * Submit the review; the system automatically triggers the BERT classification pipeline, showing the predicted label (e.g., *Positive*) and scoring details in a banner.
5. **AI Chatbot**:
   * Click the floating chat circle (bottom-right).
   * Select a chip like **Track order** or **Log Complaint**.
   * Type search keywords like `find iPhone` or file complaints like `register a complaint: Delayed shipping on my headphones` to see the automated database responses.

### B. Testing Company Portal
1. **Log In**: Sign in with a registered corporate account:
   * **Email**: `apple@company.com`
   * **Password**: `company123`
2. **Platform Dashboards**: Check the stats counter (Revenue, Retention rate, SVD accuracy). Examine the custom glowing **Sales Trend line chart** and the **Customer Segment distribution donut chart**.
3. **Reports Center**: Click **Download PDF** or **Export Excel** in the top action header to download analytics summaries.
4. **Counterfeit Verification Scanner**:
   * Navigate to **Counterfeit Detection**.
   * Enter a brand (e.g. `Nike`) and select a product SKU.
   * Select a logo file and a packaging package image.
   * Click **Execute AI Mismatch Check**.
   * The system aligns the images, calculates similarity margins, outputs genuine/fake decisions, and renders a red-highlighted differencing overlay isolating anomalous packaging regions.
