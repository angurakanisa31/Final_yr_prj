import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class SVDRecommender:
    def __init__(self, n_factors=5):
        self.n_factors = n_factors
        self.user_mapper = {}
        self.item_mapper = {}
        self.user_inv_mapper = {}
        self.item_inv_mapper = {}
        self.U = None
        self.sigma = None
        self.Vt = None
        self.user_ratings_mean = None
        self.ratings_matrix_filled = None
        self.products_df = None

    def fit(self, ratings_df: pd.DataFrame, products_df: pd.DataFrame):
        """
        Fits SVD on ratings dataframe with columns: [customer_id, product_id, rating]
        products_df contains columns: [id, name, category, brand_name]
        """
        self.products_df = products_df.copy()

        if ratings_df.empty:
            return self

        # Generate unique mapping indices
        unique_users = ratings_df["customer_id"].unique()
        unique_items = ratings_df["product_id"].unique()

        self.user_mapper = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_mapper = {iid: idx for idx, iid in enumerate(unique_items)}
        self.user_inv_mapper = {idx: uid for uid, idx in self.user_mapper.items()}
        self.item_inv_mapper = {idx: iid for iid, idx in self.item_mapper.items()}

        # Create user-item interaction matrix
        num_users = len(unique_users)
        num_items = len(unique_items)
        
        R = np.zeros((num_users, num_items))
        for _, row in ratings_df.iterrows():
            u_idx = self.user_mapper[row["customer_id"]]
            i_idx = self.item_mapper[row["product_id"]]
            R[u_idx, i_idx] = row["rating"]

        # De-mean ratings
        self.user_ratings_mean = np.mean(R, axis=1).reshape(-1, 1)
        R_demeaned = R - self.user_ratings_mean

        # Apply SVD
        # Use min(n_factors, rank - 1)
        k = min(self.n_factors, min(R_demeaned.shape) - 1)
        if k >= 1:
            try:
                # Use scipy SVD
                u, s, vt = svds(R_demeaned, k=k)
                # Sort ascending, reverse to get descending order of singular values
                idx = np.argsort(s)[::-1]
                self.U = u[:, idx]
                self.sigma = np.diag(s[idx])
                self.Vt = vt[idx, :]
                
                # Reconstructed ratings
                self.ratings_matrix_filled = np.dot(np.dot(self.U, self.sigma), self.Vt) + self.user_ratings_mean
            except Exception:
                # Fallback to mean rating or simple numpy svd
                self.ratings_matrix_filled = R
        else:
            self.ratings_matrix_filled = R

        return self

    def recommend_personalized(self, customer_id: int, top_n=10) -> list[int]:
        """Collaborative Filtering (SVD) Personalized Recommendations."""
        if customer_id not in self.user_mapper or self.ratings_matrix_filled is None:
            # Fallback to trending if user is not in database (cold start)
            return self.get_trending_products(top_n)

        user_idx = self.user_mapper[customer_id]
        user_predictions = self.ratings_matrix_filled[user_idx]

        # Get sorted predictions
        sorted_indices = np.argsort(user_predictions)[::-1]
        
        recs = []
        for idx in sorted_indices:
            item_id = self.item_inv_mapper[idx]
            recs.append(int(item_id))
            if len(recs) >= top_n:
                break
        return recs

    def get_similar_products(self, product_id: int, top_n=10) -> list[int]:
        """Item-Item collaborative similarity using SVD embeddings or simple category matching."""
        if product_id not in self.item_mapper or self.Vt is None:
            # Fallback: find products in same category
            return self.get_similar_by_category(product_id, top_n)

        # Vector representation of item is the column in Vt (i.e. row in V)
        item_idx = self.item_mapper[product_id]
        item_vector = self.Vt[:, item_idx]

        # Calculate cosine similarity with all items
        norms = np.linalg.norm(self.Vt, axis=0)
        norms[norms == 0] = 1e-9
        
        sims = np.dot(self.Vt.T, item_vector) / (norms * np.linalg.norm(item_vector))
        
        # Sort and exclude current item
        sorted_indices = np.argsort(sims)[::-1]
        recs = []
        for idx in sorted_indices:
            iid = self.item_inv_mapper[idx]
            if iid != product_id:
                recs.append(int(iid))
            if len(recs) >= top_n:
                break
        return recs

    def get_similar_by_category(self, product_id: int, top_n=10) -> list[int]:
        """Fallback method to retrieve similar items in the same category."""
        if self.products_df is None or self.products_df.empty:
            return []
        
        current_prod = self.products_df[self.products_df["id"] == product_id]
        if current_prod.empty:
            return list(self.products_df["id"].head(top_n).values)
            
        category = current_prod.iloc[0]["category"]
        similar_items = self.products_df[
            (self.products_df["category"] == category) & 
            (self.products_df["id"] != product_id)
        ]
        
        return [int(x) for x in similar_items["id"].head(top_n).values]

    def get_trending_products(self, top_n=10) -> list[int]:
        """Overall popular products across the platform."""
        if self.products_df is None or self.products_df.empty:
            return []
        # Return first top_n products (or sort by id for consistency if ratings are sparse)
        return [int(x) for x in self.products_df["id"].head(top_n).values]

    def get_cross_selling_products(self, cart_item_ids: list[int], top_n=10) -> list[int]:
        """Recommend complementary items that belong to different categories than cart items."""
        if self.products_df is None or self.products_df.empty:
            return []
            
        cart_categories = set()
        if cart_item_ids:
            cart_categories = set(self.products_df[self.products_df["id"].isin(cart_item_ids)]["category"].values)
            
        # Recommend from other categories
        cross_items = self.products_df[~self.products_df["category"].isin(cart_categories)]
        if cross_items.empty:
            cross_items = self.products_df[~self.products_df["id"].isin(cart_item_ids)]
            
        return [int(x) for x in cross_items["id"].head(top_n).values]
        
    def get_frequently_purchased(self, purchases_df: pd.DataFrame, customer_id: int, top_n=10) -> list[int]:
        """Get product IDs that a specific user has purchased multiple times."""
        if purchases_df.empty:
            return []
        user_purchases = purchases_df[purchases_df["customer_id"] == customer_id]
        if user_purchases.empty:
            return []
        # Group by product_id and sort by count
        freq = user_purchases.groupby("product_id").size().reset_index(name="counts")
        freq = freq.sort_values(by="counts", ascending=False)
        return [int(x) for x in freq["product_id"].head(top_n).values]
